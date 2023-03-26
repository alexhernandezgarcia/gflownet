"""
An environment inspired by the game of Tetris.
"""
import itertools
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.tetris.constants import PIECES


class Tetris(GFlowNetEnv):
    """
    Tetris environment: an environment inspired by the game of tetris. It's not
    supposed to be a game, but rather a toy environment with an intuitive state and
    action space.

    The state space is 2D board, with all the combinations of pieces on it.

    The action space is the choice of piece, its rotation and horizontal location
    where to drop the piece. The action space may be constrained according to needs.

    Attributes
    ----------
    width : int
        Width of the board.

    height : int
        Height of the board.

    pieces : list
        Pieces to use, identified by [I, J, L, O, S, T, Z]

    rotations : list
        Valid rotations, from [0, 90, 180, 270]
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        pieces: List = ["I", "J", "L", "O", "S", "T", "Z"],
        rotations: List = [0, 90, 180, 270],
        allow_redundant_rotations: bool = False,
        allow_eos_before_full: bool = False,
        **kwargs,
    ):
        assert width > 3
        assert height > 4
        assert all([p in ["I", "J", "L", "O", "S", "T", "Z"] for p in pieces])
        assert all([r in [0, 90, 180, 270] for r in rotations])
        self.width = width
        self.height = height
        self.pieces = pieces
        self.rotations = rotations
        self.allow_redundant_rotations = allow_redundant_rotations
        self.allow_eos_before_full = allow_eos_before_full
        # Helper functions and dicts
        self.piece2idx = lambda letter: PIECES[letter][0]
        self.piece2mat = lambda letter: torch.tensor(
            PIECES[letter][1], dtype=torch.uint8
        )
        self.rot2idx = {0: 0, 90: 1, 180: 2, 270: 3}
        # Source state: empty board
        self.source = torch.zeros((self.height, self.width), dtype=torch.uint8)
        # End-of-sequence action: all -1
        self.eos = (-1, -1, -1)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos. An action is
        represented by a tuple of length 3 (piece, rotation, location). The piece is
        represented by its index, the rotation by the integer rotation in degrees
        and the location by horizontal cell in the board of the left-most part of the
        piece.
        """
        actions = []
        pieces_mat = []
        for piece in self.pieces:
            for rotation in self.rotations:
                piece_mat = torch.rot90(self.piece2mat(piece), k=self.rot2idx[rotation])
                if self.allow_redundant_rotations or not any(
                    [torch.equal(p, piece_mat) for p in pieces_mat]
                ):
                    pieces_mat.append(piece_mat)
                else:
                    continue
                for location in range(self.width):
                    if location + piece_mat.shape[1] <= self.width:
                        actions.append((self.piece2idx(piece), rotation, location))
        actions.append(self.eos)
        return actions

    def _drop_piece_on_board(
        self, action, state: Optional[TensorType["height", "width"]] = None
    ):
        """
        Drops a piece defined by the argument action onto the board. It returns an
        updated board (copied) and a boolean variable, which is True if the piece can
        be dropped onto the current and False otherwise.
        """
        if state is None:
            state = self.state.clone().detach()
        board = state.clone().detach()
        piece_idx, rotation, location = action
        piece_mat = torch.rot90(
            self.piece2mat(self.pieces[piece_idx - 1]), k=self.rot2idx[rotation]
        )
        hp, wp = piece_mat.shape
        for row in reversed(range(self.height)):
            if row - hp + 1 < 0:
                return board, False
            if location + wp > self.width:
                return board, False
            board_section = board[row - hp + 1 : row + 1, location : location + wp]
            if sum(board_section[piece_mat != 0]) == 0:
                board[row - hp + 1 : row + 1, location : location + wp] += piece_mat
                return board, True

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, action in enumerate(self.action_space[:-1]):
            _, valid = self._drop_piece_on_board(action)
            if not valid:
                mask[idx] = True
        if not self.allow_eos_before_full and not all(mask[:-1]):
            mask[-1] = True
        return mask

    def state2oracle(
        self, state: Optional[TensorType["height", "width"]] = None
    ) -> TensorType["height", "width"]:
        """
        Prepares a state in "GFlowNet format" for the oracles: simply converts non-zero
        (non-empty) cells into 1s.

        Args
        ----
        state : tensor
        """
        if state is None:
            state = self.state.clone().detach()
        state_oracle = state.clone().detach()
        state_oracle[state_oracle != 0] = 1
        return state_oracle

    def statebatch2oracle(
        self, states: List[TensorType["height", "width"]]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracles: simply
        converts non-zero (non-empty) cells into 1s.

        Args
        ----
        state : list
        """
        states = torch.stack(states)
        states[states != 0] = 1
        return states

    def statetorch2oracle(
        self, states: TensorType["height", "width", "batch"]
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracles: : simply
        converts non-zero (non-empty) cells into 1s.
        """
        states[states != 0] = 1
        return states

    def state2policy(
        self, state: Optional[TensorType["height", "width"]] = None
    ) -> TensorType["height", "width"]:
        """
        Prepares a state in "GFlowNet format" for the policy model.

        See: state2oracle()
        """
        return self.state2oracle(state)

    def statebatch2policy(
        self, states: List[TensorType["height", "width"]]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the policy model.

        See statebatch2oracle().
        """
        return self.statebatch2oracle(states)

    def statetorch2policy(
        self, states: TensorType["height", "width", "batch"]
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "GFlowNet format" for the policy model.

        See statetorch2oracle().
        """
        return self.statetorch2oracle(states)

    def state2readable(self, state):
        """
        Converts a state (board) into a human-friendly string.
        """
        readable = str(state.numpy())
        readable = readable.replace("[[", "[").replace("]]", "]").replace("\n ", "\n")
        return readable

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        state = []
        rows = readable.split("\n")
        for row in rows:
            state.append(
                torch.tensor(
                    [int(el) for el in row.strip("[]").split(" ")], dtype=torch.uint8
                )
            )
        return torch.stack(state)

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
            for idx, action in enumerate(self.action_space[:-1]):
                parent = state.copy()
                for dim, incr in enumerate(action):
                    if parent[dim] - incr >= 0:
                        parent[dim] -= incr
                    else:
                        break
                else:
                    parents.append(parent)
                    actions.append(action)
        return parents, actions

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # If done, return invalid
        if self.done:
            return self.state, action, False
        # If action not found in action space raise an error
        if action not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        else:
            action_idx = self.action_space.index(action)
        # If action is in invalid mask, return invalid
        mask_invalid = self.get_mask_invalid_actions_forward()
        if mask_invalid[action_idx]:
            return self.state, action, False
        # If only possible action is eos, then force eos
        if all(mask_invalid[:-1]):
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        else:
            state_next, valid = self._drop_piece_on_board(action)
            if valid:
                self.state = state_next
                self.n_actions += 1
            return self.state, action, valid

    # TODO
    def get_max_traj_length(self):
        return 1e9
