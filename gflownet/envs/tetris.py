"""
An environment inspired by the game of Tetris.
"""
import itertools
import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device, tint

PIECES = {
    "I": [1, [[1], [1], [1], [1]]],
    "J": [2, [[0, 2], [0, 2], [2, 2]]],
    "L": [3, [[3, 0], [3, 0], [3, 3]]],
    "O": [4, [[4, 4], [4, 4]]],
    "S": [5, [[0, 5, 5], [5, 5, 0]]],
    "T": [6, [[6, 6, 6], [0, 6, 0]]],
    "Z": [7, [[7, 7, 0], [0, 7, 7]]],
}


class Tetris(GFlowNetEnv):
    """
    Tetris environment: an environment inspired by the game of tetris. It's not
    supposed to be a game, but rather a toy environment with an intuitive state and
    action space.

    The state space is 2D board, with all the combinations of pieces on it. Pieces that
    are added to the board are identified by a number that starts from
    piece_idx * max_pieces_per_type, and is incremented by 1 with each new piece from
    the same type. This number fills in the cells of the board where the piece is
    located. This enables telling apart pieces of the same type.

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
        assert all([p in ["I", "J", "L", "O", "S", "T", "Z"] for p in pieces])
        assert all([r in [0, 90, 180, 270] for r in rotations])
        self.device = set_device(kwargs["device"])
        self.int = torch.int16
        self.width = width
        self.height = height
        self.pieces = pieces
        self.rotations = rotations
        self.allow_redundant_rotations = allow_redundant_rotations
        self.allow_eos_before_full = allow_eos_before_full
        self.max_pieces_per_type = 100
        # Helper functions and dicts
        self.piece2idx = lambda letter: PIECES[letter][0]
        self.idx2piece = {v[0]: k for k, v in PIECES.items()}
        self.piece2mat = lambda letter: tint(
            PIECES[letter][1], int_type=self.int, device=self.device
        )
        self.rot2idx = {0: 0, 90: 1, 180: 2, 270: 3}
        # Check width and height compatibility
        heights, widths = [], []
        for piece in self.pieces:
            for rotation in self.rotations:
                piece_mat = torch.rot90(self.piece2mat(piece), k=self.rot2idx[rotation])
                hp, wp = piece_mat.shape
                heights.append(hp)
                widths.append(wp)
        assert all([self.height >= h for h in widths])
        assert all([self.width >= w for w in widths])
        # Source state: empty board
        self.source = torch.zeros(
            (self.height, self.width), dtype=self.int, device=self.device
        )
        # End-of-sequence action: all -1
        self.eos = (-1, -1, -1)
        # Conversions
        self.state2proxy = self.state2oracle
        self.statebatch2proxy = self.statebatch2oracle
        self.statetorch2proxy = self.statetorch2oracle

        # Precompute all possible rotations of each piece and the corresponding binary
        # mask
        self.piece_rotation_mat = {}
        self.piece_rotation_mask_mat = {}
        for p in pieces:
            self.piece_rotation_mat[p] = {}
            self.piece_rotation_mask_mat[p] = {}
            for r in rotations:
                self.piece_rotation_mat[p][r] = torch.rot90(
                    self.piece2mat(p), k=self.rot2idx[r]
                )
                self.piece_rotation_mask_mat[p][r] = self.piece_rotation_mat[p][r] != 0

        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos. An action is
        represented by a tuple of length 3 (piece, rotation, col). The piece is
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
                for col in range(self.width):
                    if col + piece_mat.shape[1] <= self.width:
                        actions.append((self.piece2idx(piece), rotation, col))
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

        piece_idx, rotation, col = action
        piece_mat = self.piece_rotation_mat[self.idx2piece[piece_idx]][rotation]
        piece_mat_mask = self.piece_rotation_mask_mat[self.idx2piece[piece_idx]][
            rotation
        ]
        hp, wp = piece_mat.shape

        # Check if piece goes overboard horizontally
        if col + wp > self.width:
            return board, False

        # Find the highest row occupied by any other piece in the columns where we wish
        # to add the new piece
        occupied_rows = board[:, col : col + wp].sum(1).nonzero()
        if len(occupied_rows) == 0:
            # All rows are unoccupied. Set highest occupied row as the row "below" the
            # board.
            highest_occupied_row = self.height
        else:
            highest_occupied_row = occupied_rows[0, 0]

        # Iteratively attempt to place piece on the board starting from the top.
        # As soon as we reach a row where we can't place the piece because it
        # creates a collision, we can stop the search (since we can't put a piece under
        # this obstacle anyway).
        starting_row = highest_occupied_row - hp
        lowest_valid_row = None
        for row in range(starting_row, self.height - hp + 1):
            if row == -hp:
                # Placing the piece here would make it land fully outside the board.
                # This means that there is no place on the board for the piece
                break

            elif row < 0:
                # It is not possible to place the piece at this row because the piece
                # would not completely be in the board. However, it is still possible
                # to check for obstacles because any obstacle will prevent placing the
                # piece at any position below
                board_section = board[: row + hp, col : col + wp]
                piece_mask_section = piece_mat_mask[-(row + hp) :]
                if (board_section * (piece_mask_section != 0)).any():
                    # An obstacle has been found.
                    break

            else:
                # The piece can be placed here if all board cells under piece are empty
                board_section = board[row : row + hp, col : col + wp]
                if (board_section * piece_mat_mask).any():
                    # The piece cannot be placed here and cannot be placed any lower
                    # because of an obstacle.
                    break
                else:
                    # The piece can be placed here.
                    lowest_valid_row = row

        # Place the piece if possible
        if lowest_valid_row is None:
            # The piece cannot be placed. Return the board as-is.
            return board, False
        else:
            # Get and set index of new piece
            piece_idx = self._get_max_piece_idx(board, piece_idx, incr=1)
            piece_mat[piece_mat_mask] = piece_idx

            # Place the piece on the board.
            row = lowest_valid_row
            board[row : row + hp, col : col + wp] += piece_mat
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
            _, valid = self._drop_piece_on_board(action, state)
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
        return self.state2oracle(state).flatten()

    def statebatch2policy(
        self, states: List[TensorType["height", "width"]]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the policy model.

        See statebatch2oracle().
        """
        return self.statebatch2oracle(states).flatten(start_dim=1)

    def statetorch2policy(
        self, states: TensorType["height", "width", "batch"]
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "GFlowNet format" for the policy model.

        See statetorch2oracle().
        """
        return self.statetorch2oracle(states).flatten(start_dim=1)

    def policy2state(
        self, policy: Optional[TensorType["height", "width"]] = None
    ) -> TensorType["height", "width"]:
        """
        Returns None to signal that the conversion is not reversible.

        See: state2oracle()
        """
        return None

    def state2readable(self, state: Optional[TensorType["height", "width"]] = None):
        """
        Converts a state (board) into a human-friendly string.
        """
        state = self._get_state(state)
        if isinstance(state, tuple):
            readable = str(np.stack(state))
        else:
            readable = str(state.cpu().numpy())
        readable = readable.replace("[[", "[").replace("]]", "]").replace("\n ", "\n")
        return readable

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        pattern = re.compile(r"\s+")
        state = []
        rows = readable.split("\n")
        for row in rows:
            # Preprocess
            row = re.sub(pattern, " ", row)
            row = row.replace(" ]", "]")
            row = row.replace("[ ", "[")
            # Process
            state.append(
                tint(
                    [int(el) for el in row.strip("[]").split(" ")],
                    int_type=self.int,
                    device=self.device,
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

        See: _is_parent_action()

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
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
            indices = state.unique()
            for idx in indices[indices > 0]:
                if self._piece_can_be_lifted(state, idx):
                    piece_idx, rotation, col = self._get_idx_rotation_col(state, idx)
                    parent = state.clone().detach()
                    parent[parent == idx] = 0
                    action = (piece_idx, rotation, col)
                    parents.append(parent)
                    actions.append(action)
        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        # If action is eos
        if action == self.eos:
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
        return int(1e9)

    def set_state(
        self, state: TensorType["height", "width"], done: Optional[bool] = False
    ):
        """
        Sets the state and done. If done is True but incompatible with state (done is
        True, allow_eos_before_full is False and state is not full), then force done
        False and print warning. Also, make sure state is tensor.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.int16)
        if done is True and not self.allow_eos_before_full:
            mask = self.get_mask_invalid_actions_forward(state, done=False)
            if not all(mask[:-1]):
                done = False
                warnings.warn(
                    f"Attempted to set state\n\n{self.state2readable(state)}\n\n"
                    "with done = True, which is not compatible with "
                    "allow_eos_before_full = False. Forcing done = False."
                )
        self.state = state
        self.done = done
        return self

    def _piece_can_be_lifted(self, board, piece_idx):
        """
        Returns True if the piece with index piece_idx could be lifted, that is all
        cells of the board above the piece are zeros. False otherwise.
        """
        board_aux = board.clone().detach()
        if piece_idx < self.max_pieces_per_type:
            piece_idx = self._get_max_piece_idx(board_aux, piece_idx, incr=0)
        rows, cols = torch.where(board_aux == piece_idx)
        board_top = torch.cat([board[:r, c] for r, c in zip(rows, cols)])
        board_top[board_top == piece_idx] = 0
        return not any(board_top)

    def _get_idx_rotation_col(self, board, piece_idx):
        piece_idx_base = int(piece_idx / self.max_pieces_per_type)
        board_aux = board.clone().detach()
        piece_mat = self.piece2mat(self.idx2piece[piece_idx_base])
        rows, cols = torch.where(board_aux == piece_idx)
        row = min(rows).item()
        col = min(cols).item()
        hp = max(rows).item() - row + 1
        wp = max(cols).item() - col + 1
        board_section = board_aux[row : row + hp, col : col + wp]
        board_section[board_section != piece_idx] = 0
        board_section[board_section == piece_idx] = piece_idx_base
        for rotation in self.rotations:
            piece_mat_rot = torch.rot90(piece_mat, k=self.rot2idx[rotation])
            if piece_mat_rot.shape == board_section.shape and torch.equal(
                torch.rot90(piece_mat, k=self.rot2idx[rotation]), board_section
            ):
                return piece_idx_base, rotation, col
        raise ValueError(
            f"No valid rotation found for piece {piece_idx} in board {board}"
        )

    def _get_max_piece_idx(
        self, board: TensorType["height", "width"], piece_idx: int, incr: int = 0
    ):
        """
        Gets the index of a new piece with base index piece_idx, based on the board.

        board : tensor
            The current board matrix.

        piece_idx : int
            Piece index, in base format [1, 2, ...]

        incr : int
            Increment of the returned index with respect to the max.
        """

        min_idx = piece_idx * self.max_pieces_per_type
        max_idx = min_idx + self.max_pieces_per_type
        max_relevant_piece_idx = (board * (board < max_idx)).max()

        if max_relevant_piece_idx >= min_idx:
            return max_relevant_piece_idx + incr
        else:
            return min_idx

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None, n_factor_max: int = 10
    ) -> List[List]:
        rng = np.random.default_rng(seed)
        n_iter_max = n_states * n_factor_max
        states = []
        for it in range(int(n_iter_max)):
            self.reset()
            while not self.done:
                # Sample random action
                mask_invalid = torch.unsqueeze(
                    torch.BoolTensor(self.get_mask_invalid_actions_forward()), 0
                )
                random_policy = torch.unsqueeze(
                    torch.tensor(self.random_policy_output, dtype=self.float), 0
                )
                actions, _ = self.sample_actions(
                    policy_outputs=random_policy, mask_invalid_actions=mask_invalid
                )
                _, _, _ = self.step(actions[0])
            if not any([torch.equal(self.state, s) for s in states]):
                states.append(self.state)
            if len(states) == n_states:
                break
        return states
