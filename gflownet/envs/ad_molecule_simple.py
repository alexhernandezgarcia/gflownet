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
        self.conformer = Conformer(atom_positions, constants.ad_smiles, constants.ad_atom_types)
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
        graph =  self.conformer.dgl_graph
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
        if device == torch.device('cpu'):
            np_states = states.numpy()
        else: 
            np_states = states.cpu().numpy()

        state_policy_batch = dgl.batch([self.state2policy(st) for st in np_states]).to(device)
        return state_policy_batch

    def reset(self):
        super().reset()
        new_atom_pos = self.atom_positions_dataset.sample()
        self.set_condition(new_atom_pos)
        return self

    def get_fixed_policy_output(self):
        #todo
        pass

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        random_action_prob=0.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Random actions
        n_random = int(n_states * random_action_prob)
        idx_random = torch.randint(high=n_states, size=(n_random,))
        policy_outputs[idx_random, :] = torch.tensor(self.fixed_policy_output).to(
            policy_outputs
        )
        # Sample dimensions
        if sampling_method == "uniform":
            logits_dims = torch.ones(n_states, self.policy_output_dim).to(device)
        elif sampling_method == "policy":
            logits_dims = policy_outputs[:, 0::3]
            logits_dims /= temperature_logits
        if mask_invalid_actions is not None:
            logits_dims[mask_invalid_actions] = -loginf
        dimensions = Categorical(logits=logits_dims).sample()
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Sample angle increments
        ns_range_noeos = ns_range[dimensions != self.eos]
        dimensions_noeos = dimensions[dimensions != self.eos]
        angles = torch.zeros(n_states).to(device)
        logprobs_angles = torch.zeros(n_states).to(device)
        if len(dimensions_noeos) > 0:
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    2 * torch.pi * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                locations = policy_outputs[:, 1::3][ns_range_noeos, dimensions_noeos]
                concentrations = policy_outputs[:, 2::3][
                    ns_range_noeos, dimensions_noeos
                ]
                distr_angles = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_concentration_epsilon,
                )
            angles[ns_range_noeos] = distr_angles.sample()
            logprobs_angles[ns_range_noeos] = distr_angles.log_prob(
                angles[ns_range_noeos]
            )
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_angles
        # Build actions
        actions = [
            (dimension, angle)
            for dimension, angle in zip(dimensions.tolist(), angles.tolist())
        ]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", 2],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        device = policy_outputs.device
        dimensions, angles = zip(*actions)
        dimensions = torch.LongTensor([d.long() for d in dimensions]).to(device)
        angles = torch.FloatTensor(angles).to(device)
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Dimensions
        logits_dims = policy_outputs[:, 0::3]
        if mask_invalid_actions is not None:
            logits_dims[mask_invalid_actions] = -loginf
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Angle increments
        # Cases where only one angle transition is possible (logp(angle) = 1):
        # - (A: Number of dimensions different to source == number of states, and
        # - B: Angle of selected dimension == source), or
        # - C: Dimension == eos
        # Cases where angles should be sampled from the distribution:
        # ~((A & B) | C) = ~(A & B) & ~C = (~A | ~B) & ~C
        source = torch.tensor(self.source, device=device)
        source_aux = torch.tensor(self.source + [-1], device=device)
        nsource_ne_nsteps = torch.ne(
            torch.sum(torch.ne(states_target[:, :-1], source), axis=1),
            states_target[:, -1],
        )
        angledim_ne_source = torch.ne(
            states_target[ns_range, dimensions], source_aux[dimensions]
        )
        noeos = torch.ne(dimensions, self.eos)
        nofix_indices = torch.logical_and(
            torch.logical_or(nsource_ne_nsteps, angledim_ne_source),
            noeos,
        )
        ns_range_nofix = ns_range[nofix_indices]
        dimensions_nofix = dimensions[nofix_indices]
        logprobs_angles = torch.zeros(n_states).to(device)
        if len(dimensions_nofix) > 0:
            locations = policy_outputs[:, 1::3][ns_range_nofix, dimensions_nofix]
            concentrations = policy_outputs[:, 2::3][ns_range_nofix, dimensions_nofix]
            distr_angles = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_concentration_epsilon,
            )
            logprobs_angles[ns_range_nofix] = distr_angles.log_prob(
                angles[ns_range_nofix]
            )
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_angles
        return logprobs