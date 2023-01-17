import numpy as np

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.atom_positions_dataset import AtomPositionsDataset
from gflownet.utils.molecule.conformer import Conformer 

class ADMolecule(ContinuousTorus):
    def __init__(
        self,
        path_to_dataset,
        length_traj=1,
        vonmises_mean=0.0,
        vonmises_concentration=0.5,
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
        self.conformer = Conformer(atom_positions, constants.ad_smiles, constants.ad_atom_types)
        n_dim = len(self.conformer.freely_rotatable_tas)
        super(ADMolecule, self).__init__(
            n_dim,
            length_traj,
            vonmises_mean,
            vonmises_concentration,
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            denorm_proxy,
            energies_stats,
            proxy,
            oracle,
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
        graph =  self.conformer.dgl_graph
        n_atoms = self.conformer.get_n_atoms()
        step_feature = torch.ones(n_atoms, 1) * state[-1]
        graph.ndata[constants.step_feature_name] = step_feature

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

    def reset(self):
        super().reset()
        new_atom_pos = self.atom_positions_dataset.sample()
        self.set_condition(new_atom_pos)
        return self