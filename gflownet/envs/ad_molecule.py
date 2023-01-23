import dgl
import numpy as np

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.atom_positions_dataset import AtomPositionsDataset
from gflownet.utils.molecule.conformer import Conformer
from gflownet.utils.molecule.distributions import get_mixture_of_projected_normals


class ADMolecule(ContinuousTorus):
    def __init__(
        self,
        path_to_dataset,
        length_traj=1,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="boltzmann",
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle=None,
        **kwargs,
    ):
        self.atom_positions_dataset = AtomPositionsDataset(path_to_dataset)
        atom_positions = self.atom_positions_dataset.sample()
        self.conformer = Conformer(
            atom_positions, constants.ad_smiles, constants.ad_atom_types
        )
        n_dim = len(self.conformer.freely_rotatable_tas)
        super(ADMolecule, self).__init__(
            n_dim=n_dim,
            length_traj=length_traj,
            env_id=env_id,
            reward_beta=reward_beta,
            reward_norm=reward_norm,
            reward_norm_std_mult=reward_norm_std_mult,
            reward_func=reward_func,
            denorm_proxy=denorm_proxy,
            energies_stats=energies_stats,
            proxy=proxy,
            oracle=oracle,
            **kwargs,
        )
        self.sync_conformer_with_state()

    def sync_conformer_with_state(self, state: List = None):
        if state is None:
            state = self.state
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])

    def set_condition(self, atom_positions):
        """
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions
        """
        self.conformer.set_atom_positions(atom_positions)
        self.sync_conformer_with_state()

    def state2policy(self, state: List = None) -> "dgl graph (w/ time)":
        self.sync_conformer_with_state(state)
        graph = self.conformer.dgl_graph
        n_atoms = self.conformer.get_n_atoms()
        step_feature = torch.ones(n_atoms, 1) * state[-1]
        graph.ndata[constants.step_feature_name] = step_feature
        return graph

    def policy2state(self, state_policy: "dgl graph (w/ time)") -> List:
        positions = state_policy.ndata[constants.atom_position_name].numpy()
        self.conformer.set_atom_positions(positions)
        angles = self.conformer.get_freely_rotatable_ta_values()
        step = state_policy.ndata[constants.step_feature_name][0][0]
        return [*angles, step]

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray[np.float32]:
        # todo
        """
        Prepares a batch of states in "GFlowNet format" for the proxy: an array where
        each state is a row of length n_dim with an angle in radians. The n_actions
        item is removed.
        """
        return np.array(states)[:, :-1]

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> "dgl Graph with a batch of molecules inside":
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        device = states.device
        if device == torch.device("cpu"):
            np_states = states.numpy()
        else:
            np_states = states.cpu().numpy()

        state_policy_batch = dgl.batch([self.state2policy(st) for st in np_states]).to(
            device
        )
        return state_policy_batch

    def reset(self):
        super().reset()
        new_atom_pos = self.atom_positions_dataset.sample()
        self.set_condition(new_atom_pos)
        return self
