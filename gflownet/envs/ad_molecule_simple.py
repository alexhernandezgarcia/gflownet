import dgl
import numpy as np
import numpy.typing as npt

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.atom_positions_dataset import AtomPositionsDataset
from gflownet.utils.molecule.conformer_base import ConformerBase 

class ADMoleculeSimple(ContinuousTorus):
    """Simple extension of 2d continuous torus where reward function is defined by the 
    energy of the alanine dipeptide molecule"""  
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
        self.conformer = ConformerBase(atom_positions, constants.ad_smiles, 
                                       constants.ad_atom_types, constants.ad_free_tas)
        n_dim = len(self.conformer.freely_rotatable_tas)
        super(ADMoleculeSimple, self).__init__(
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
        return self.conformer

    def set_condition(self, atom_positions):
        """
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions
        """
        self.conformer.set_atom_positions(atom_positions)
        self.sync_conformer_with_state()

    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> List[Tuple[npt.NDArray, npt.NDArray]]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the proxy.
        """
        device = states.device
        if device == torch.device('cpu'):
            np_states = states.numpy()
        else: 
            np_states = states.cpu().numpy()
        states_proxy = []
        for st in np_states:
            conf = self.sync_conformer_with_state(st)
            states_proxy.append((conf.get_atomic_numbers(), conf.get_atom_positions()))
        return states_proxy

    def reset(self):
        super().reset()
        # no resets of the condition, keep it simple
        return self