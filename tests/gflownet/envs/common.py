"""
Tests common to all environments.

Note that copying the state is necessary in some tests to preserve specific values of
the state. This is necessary because in some environments the state is a list which is
updated when an action is applied. Therefore, if the tests needs to keep older values
of the state, for example in test__trajectories_are_reversible(), a copy is needed.
"""

import inspect
import warnings

import hydra
import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize
from utils_for_tests import load_base_test_config

from gflownet.utils.common import copy, gflownet_from_config, tbool, tfloat
from gflownet.utils.policy import parse_policy_config


class BaseTestsCommon:
    def test__set_state__sets_expected_state_and_creates_copy(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            states = self.env.get_random_states(n_states)
            envs = []
            for state in states:
                for idx in range(5):
                    env_new = self.env.copy().reset(idx)
                    env_new.set_state(state, done=False)
                    assert self.env.equal(state, env_new.state)
                    envs.append(env_new)
            state_ids = [id(env.state) for env in envs]
            assert len(np.unique(state_ids)) == len(state_ids)

    def test__set_state__terminating_sets_expected_state_and_creates_copy(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            states = _get_terminating_states(self.env, n_states)
            envs = []
            for state in states:
                for idx in range(5):
                    env_new = self.env.copy().reset(idx)
                    env_new.set_state(state, done=True)
                    assert self.env.equal(state, env_new.state)
                    envs.append(env_new)
            state_ids = [id(env.state) for env in envs]
            assert len(np.unique(state_ids)) == len(state_ids)

    def test__step__returns_same_state_action_and_invalid_if_done(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            # Sample random trajectory
            self.env.trajectory_random()
            assert self.env.done
            # Attempt another step
            action = self.env.action_space[
                np.random.randint(low=0, high=self.env.action_space_dim)
            ]
            next_state, action_step, valid = self.env.step(action)
            if torch.is_tensor(self.env.state):
                assert self.env.equal(next_state, self.env.state)
            else:
                assert next_state == self.env.state
            assert action_step == action
            assert valid is False

    def test__sample_actions__backward__returns_eos_if_done(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            # Set states, done and get masks
            masks = []
            for state in states:
                self.env.set_state(state, done=True)
                masks.append(self.env.get_mask_invalid_actions_backward())
            # Build random policy outputs and tensor masks
            policy_outputs = torch.tile(self.env.random_policy_output, (len(states), 1))
            # Add noise to policy outputs
            policy_outputs += torch.randn(policy_outputs.shape)
            masks = tbool(masks, device=self.env.device)
            actions = self.env.sample_actions_batch(
                policy_outputs, masks, states, is_backward=True
            )
            assert all([action == self.env.eos for action in actions])

    def test__get_logprobs__backward__returns_zero_if_done(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            # Set states, done and get masks
            masks = []
            for state in states:
                self.env.set_state(state, done=True)
                masks.append(self.env.get_mask_invalid_actions_backward())
            # EOS actions
            actions_eos = torch.tile(
                tfloat(self.env.eos, float_type=self.env.float, device=self.env.device),
                (len(states), 1),
            )
            # Build random policy outputs and tensor masks
            policy_outputs = torch.tile(self.env.random_policy_output, (len(states), 1))
            # Add noise to policy outputs
            policy_outputs += torch.randn(policy_outputs.shape)
            masks = tbool(masks, device=self.env.device)
            logprobs = self.env.get_logprobs(
                policy_outputs, actions_eos, masks, states, is_backward=True
            )
            assert torch.all(logprobs == 0.0)

    def test__step_random__does_not_sample_invalid_actions_forward(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                valid_actions = self.env.get_valid_actions(done=False)
                # Sample random action
                state_next, action, valid = self.env.step_random()
                assert valid
                assert self.env.action2representative(action) in valid_actions
            assert True

    def test__step_random__does_not_sample_invalid_actions_backward(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            # Sample random trajectory
            self.env.trajectory_random()
            # Sample random steps backward
            while not self.env.is_source():
                valid_actions = self.env.get_valid_actions(backward=True)
                # Sample random action backwards
                state_next, action, valid = self.env.step_random(backward=True)
                assert valid
                assert self.env.action2representative(action) in valid_actions
            assert True

    def test__get_mask__is_consistent_regardless_of_inputs(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            # Sample a random state
            state = self.env.get_random_states(1, exclude_source=True)[0]
            # Reset environment (state will be source)
            self.env.reset()
            # Obtain masks by passing state
            mask_f = self.env.get_mask_invalid_actions_forward(state, done=False)
            mask_b = self.env.get_mask_invalid_actions_backward(state, done=False)
            # Set state
            self.env.set_state(state, done=False)
            # Check that masks are the same after setting the state and obtaining them
            # without passing the state as argument
            assert mask_f == self.env.get_mask_invalid_actions_forward(done=False)
            assert mask_b == self.env.get_mask_invalid_actions_backward(done=False)

    def test__get_valid_actions__is_consistent_regardless_of_inputs(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            # Sample a random state
            state = self.env.get_random_states(1, exclude_source=True)[0]
            # Reset environment (state will be source)
            self.env.reset()
            # Obtain masks by passing state
            mask_f = self.env.get_mask_invalid_actions_forward(state, done=False)
            mask_b = self.env.get_mask_invalid_actions_backward(state, done=False)
            # Obtain valid actions by passing mask and state
            valid_actions_f_mask_state = self.env.get_valid_actions(
                mask=mask_f, state=state, done=False, backward=False
            )
            valid_actions_b_mask_state = self.env.get_valid_actions(
                mask=mask_b, state=state, done=False, backward=True
            )
            # Obtain valid actions by passing state but not mask. In this case, the
            # mask should be computed from the state inside get_valid_actions.
            valid_actions_f_state = self.env.get_valid_actions(
                state=state, done=False, backward=False
            )
            valid_actions_b_state = self.env.get_valid_actions(
                state=state, done=False, backward=True
            )
            # Check that the valid actions are the same in both cases
            assert valid_actions_f_mask_state == valid_actions_f_state
            assert valid_actions_b_mask_state == valid_actions_b_state
            # Set state
            self.env.set_state(state, done=False)
            # Obtain valid actions without passing neither the mask nor the state.
            # In this case, both the state and the mask should be computed from the
            # environment.
            valid_actions_f = self.env.get_valid_actions(done=False, backward=False)
            valid_actions_b = self.env.get_valid_actions(done=False, backward=True)
            # Check that the valid actions are still the same
            assert valid_actions_f_mask_state == valid_actions_f
            assert valid_actions_b_mask_state == valid_actions_b

    def test__forward_actions_have_nonzero_backward_prob(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            policy_random = torch.unsqueeze(self.env.random_policy_output, 0)
            while not self.env.done:
                state_next, action, valid = self.env.step_random(backward=False)
                assert valid
                # Get backward logprobs
                mask_bw = self.env.get_mask_invalid_actions_backward()
                masks = torch.unsqueeze(tbool(mask_bw, device=self.env.device), 0)
                actions_torch = torch.unsqueeze(
                    tfloat(action, float_type=self.env.float, device=self.env.device), 0
                )
                policy_outputs = policy_random.clone().detach()
                logprobs_bw = self.env.get_logprobs(
                    policy_outputs=policy_outputs,
                    actions=actions_torch,
                    mask=masks,
                    states_from=[self.env.state],
                    is_backward=True,
                )
                assert torch.isfinite(logprobs_bw)
                assert logprobs_bw > -1e6
                state_prev = copy(state_next)  # TODO: We never use this. Remove?

    def test__backward_actions_have_nonzero_forward_prob(self, n_repeat=1, n_states=3):
        method_name = _get_current_method_name()

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            policy_random = torch.unsqueeze(self.env.random_policy_output, 0)

            for state in states:
                # Reset environment and set state
                self.env.reset()
                self.env.set_state(state, done=True)
                # Sample random steps backward
                while not self.env.is_source():
                    state_next, action, valid = self.env.step_random(backward=True)
                    assert valid
                    # Get forward logprobs
                    mask_fw = self.env.get_mask_invalid_actions_forward()
                    masks = torch.unsqueeze(tbool(mask_fw, device=self.env.device), 0)
                    actions_torch = torch.unsqueeze(
                        tfloat(
                            action, float_type=self.env.float, device=self.env.device
                        ),
                        0,
                    )
                    policy_outputs = policy_random.clone().detach()
                    logprobs_fw = self.env.get_logprobs(
                        policy_outputs=policy_outputs,
                        actions=actions_torch,
                        mask=masks,
                        states_from=[self.env.state],
                        is_backward=False,
                    )
                    assert torch.isfinite(logprobs_fw)
                    assert logprobs_fw > -1e6

    def test__sample_backwards_reaches_source(self, n_repeat=1, n_states=3):
        method_name = _get_current_method_name()

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            for state in states:
                self.env.reset()
                self.env.set_state(state, done=True)
                n_actions = 0
                actions = []
                while True:
                    if self.env.is_source():
                        assert True
                        break
                    next_state, action, valid = self.env.step_random(backward=True)
                    assert valid
                    assert self.env.equal(self.env.state, next_state)
                    actions.append(action)
                    n_actions += 1
                    assert n_actions <= self.env.max_traj_length

    def test__trajectories_are_reversible(self, n_repeat=1):
        # Skip for certain environments until fixed:
        skip_envs = [
            "Tree",
        ]  # TODO: handle this using the count instead.
        if self.env.__class__.__name__ in skip_envs:
            warnings.warn("Skipping test for this specific environment.")
            return

        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()

            # Sample random forward trajectory
            states_trajectory_fw = []
            actions_trajectory_fw = []
            while not self.env.done:
                state, action, valid = self.env.step_random(backward=False)
                assert valid
                # Copy prevents mutation by next step.
                states_trajectory_fw.append(copy(state))
                actions_trajectory_fw.append(action)

            # Sample backward trajectory with actions in forward trajectory
            states_trajectory_bw = []
            actions_trajectory_bw = []
            actions_trajectory_fw_copy = actions_trajectory_fw.copy()
            while not self.env.is_source() or self.env.done:
                state, action, valid = self.env.step_backwards(
                    actions_trajectory_fw_copy.pop()
                )
                assert valid
                # Copy prevents mutation by next step.
                states_trajectory_bw.append(copy(state))
                actions_trajectory_bw.append(action)

            assert all(
                [
                    self.env.equal(s_fw, s_bw)
                    for s_fw, s_bw in zip(
                        states_trajectory_fw[:-1], states_trajectory_bw[-2::-1]
                    )
                ]
            )
            assert actions_trajectory_fw == actions_trajectory_bw[::-1]

    def test__sample_actions__get_logprobs__batched_forward_trajectories(
        self, n_repeat=1, batch_size=2
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "batch_size") and method_name in self.batch_size:
            batch_size = self.batch_size[method_name]

        for _ in range(n_repeat):
            # Make list of envs
            envs = []
            for idx in range(batch_size):
                env_aux = self.env.copy().reset(idx)
                envs.append(env_aux)

            # Iterate until envs is empty
            while envs:
                # States
                states = [env.state for env in envs]
                # Masks
                masks = tbool(
                    [env.get_mask_invalid_actions_forward() for env in envs],
                    device=self.env.device,
                )
                # Policy outputs random
                policy_outputs = torch.tile(
                    self.env.random_policy_output, (len(states), 1)
                )
                # Sample batch of actions
                actions = self.env.sample_actions_batch(
                    policy_outputs=policy_outputs,
                    mask=masks,
                    states_from=states,
                    is_backward=False,
                )
                # Logprobs
                actions_torch = torch.tensor(actions)
                logprobs_glp = self.env.get_logprobs(
                    policy_outputs=policy_outputs,
                    actions=actions_torch,
                    mask=masks,
                    states_from=states,
                    is_backward=False,
                )

                # Apply steps
                for env, action, logprob in zip(envs, actions, logprobs_glp):
                    _, action, valid = env.step(action)

                    # Action must be valid and logprob must be finite and non-zero
                    assert valid
                    assert torch.isfinite(logprob)
                    assert logprob > -1e6

                # Filter out finished trajectories
                envs = [env for env in envs if not env.done]

        assert True

    def test__sample_actions__get_logprobs__batched_backward_trajectories(
        self, n_repeat=1, batch_size=2
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "batch_size") and method_name in self.batch_size:
            batch_size = self.batch_size[method_name]

        states = _get_terminating_states(self.env, batch_size)

        for _ in range(n_repeat):
            # Make list of envs
            envs = []
            for idx, state in enumerate(states):
                env_aux = self.env.copy().reset(idx)
                env_aux.set_state(state, done=True)
                envs.append(env_aux)

            # Iterate until envs is empty
            while envs:
                # States
                states = [env.state for env in envs]
                # Masks
                masks = tbool(
                    [env.get_mask_invalid_actions_backward() for env in envs],
                    device=self.env.device,
                )
                # Policy outputs random
                policy_outputs = torch.tile(
                    self.env.random_policy_output, (len(states), 1)
                )
                # Sample batch of actions
                actions = self.env.sample_actions_batch(
                    policy_outputs=policy_outputs,
                    mask=masks,
                    states_from=states,
                    is_backward=True,
                )
                # Logprobs
                actions_torch = torch.tensor(actions)
                logprobs_glp = self.env.get_logprobs(
                    policy_outputs=policy_outputs,
                    actions=actions_torch,
                    mask=masks,
                    states_from=states,
                    is_backward=True,
                )

                # Apply steps
                for env, action, logprob in zip(envs, actions, logprobs_glp):
                    _, action, valid = env.step_backwards(action)

                    # Action must be valid and logprob must be finite and non-zero
                    assert valid
                    assert torch.isfinite(logprob)
                    assert logprob > -1e6

                # Filter out finished trajectories
                envs = [env for env in envs if not env.is_source()]

        assert True

    def test__get_logprobs__all_finite_in_accumulated_forward_trajectories(
        self, n_repeat=1, batch_size=2
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "batch_size") and method_name in self.batch_size:
            batch_size = self.batch_size[method_name]

        for _ in range(n_repeat):
            # Variables to store batches of entire trajectories
            actions_all = []
            states_from_all = []
            states_to_all = []
            masks_f_all = []
            masks_b_all = []

            # Unfold as many trajectories as the batch size
            for idx in range(batch_size):
                env = self.env.copy().reset(idx)
                while not env.done:
                    state_from = copy(env.state)
                    mask_f = env.get_mask_invalid_actions_forward(done=False)
                    # Sample random action
                    state, action, valid = env.step_random()
                    if not valid:
                        continue
                    # Add to accumulated variables
                    actions_all.append(action)
                    states_from_all.append(state_from)
                    states_to_all.append(copy(env.state))
                    masks_f_all.append(mask_f)
                    masks_b_all.append(env.get_mask_invalid_actions_backward())

            # Logprobs of all actions in the batch of trajectories
            actions_torch = torch.tensor(actions_all)
            masks_f_all = tbool(masks_f_all, device=self.env.device)
            masks_b_all = tbool(masks_b_all, device=self.env.device)
            policy_outputs = torch.tile(
                self.env.random_policy_output, (len(actions_all), 1)
            )
            logprobs_f = self.env.get_logprobs(
                policy_outputs=policy_outputs,
                actions=actions_torch,
                mask=masks_f_all,
                states_from=states_from_all,
                is_backward=False,
            )
            assert torch.all(torch.isfinite(logprobs_f))
            assert torch.all(logprobs_f > -1e6)
            logprobs_b = self.env.get_logprobs(
                policy_outputs=policy_outputs,
                actions=actions_torch,
                mask=masks_b_all,
                states_from=states_to_all,
                is_backward=True,
            )
            assert torch.all(torch.isfinite(logprobs_b))
            assert torch.all(logprobs_b > -1e6)

    def test__get_logprobs__all_finite_in_random_forward_transitions(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        for _ in range(n_repeat):
            states = self.env.get_random_states(n_states)

            # Masks
            # Important: we need to set the state of the environment before computing
            # the mask because otherwise the environment can have wrong attributes,
            # such as self.space_group
            masks = []
            for state in states:
                self.env.set_state(state, done=False)
                masks.append(self.env.get_mask_invalid_actions_forward())
            masks = tbool(masks, device=self.env.device)
            # Policy outputs random
            policy_outputs = torch.tile(self.env.random_policy_output, (len(states), 1))
            # Sample batch of actions
            actions = self.env.sample_actions_batch(
                policy_outputs=policy_outputs,
                mask=masks,
                states_from=states,
                is_backward=False,
            )
            # Logprobs forward
            actions_torch = torch.tensor(actions)
            logprobs = self.env.get_logprobs(
                policy_outputs=policy_outputs,
                actions=actions_torch,
                mask=masks,
                states_from=states,
                is_backward=False,
            )
            # Apply steps
            states_next = []
            dones_next = []
            for state, action, logprob in zip(states, actions, logprobs):
                env = self.env.copy().reset()
                env.set_state(state)
                _, _, valid = env.step(action)
                states_next.append(env.state)
                dones_next.append(env.done)

                # Action must be valid and logprob must be finite and non-zero
                assert valid
                assert torch.isfinite(logprob)
                assert logprob > -1e6

            # Logprobs backward
            masks = []
            for state, done in zip(states_next, dones_next):
                self.env.set_state(state, done=done)
                masks.append(self.env.get_mask_invalid_actions_backward())
            masks = tbool(masks, device=self.env.device)
            actions_torch = torch.tensor(actions)
            logprobs = self.env.get_logprobs(
                policy_outputs=policy_outputs[: len(actions), :],
                actions=actions_torch,
                mask=masks,
                states_from=states_next,
                is_backward=True,
            )
            assert torch.all(torch.isfinite(logprobs))
            assert torch.all(logprobs > -1e6)

    def test__get_logprobs__all_finite_in_random_backward_transitions(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        for _ in range(n_repeat):
            states = self.env.get_random_states(n_states, exclude_source=True)

            # Masks
            # Important: we need to set the state of the environment before computing
            # the mask because otherwise the environment can have wrong attributes,
            # such as self.space_group
            masks = []
            for state in states:
                self.env.set_state(state, done=False)
                masks.append(self.env.get_mask_invalid_actions_backward())
            masks = tbool(masks, device=self.env.device)
            # Policy outputs random
            policy_outputs = torch.tile(self.env.random_policy_output, (len(states), 1))
            # Sample batch of actions
            actions = self.env.sample_actions_batch(
                policy_outputs=policy_outputs,
                mask=masks,
                states_from=states,
                is_backward=True,
            )
            # Logprobs backward
            actions_torch = torch.tensor(actions)
            logprobs = self.env.get_logprobs(
                policy_outputs=policy_outputs,
                actions=actions_torch,
                mask=masks,
                states_from=states,
                is_backward=True,
            )
            # Apply steps
            states_next = []
            for state, action, logprob in zip(states, actions, logprobs):
                env = self.env.copy().reset()
                env.set_state(state)
                _, _, valid = env.step_backwards(action)
                states_next.append(env.state)

                # Action must be valid and logprob must be finite and non-zero
                assert valid
                assert torch.isfinite(logprob)
                assert logprob > -1e6

            # Logprobs forward
            masks = []
            for state in states_next:
                self.env.set_state(state, done=False)
                masks.append(self.env.get_mask_invalid_actions_forward())
            masks = tbool(masks, device=self.env.device)
            actions_torch = torch.tensor(actions)
            logprobs = self.env.get_logprobs(
                policy_outputs=policy_outputs,
                actions=actions_torch,
                mask=masks,
                states_from=states_next,
                is_backward=False,
            )
            assert torch.all(torch.isfinite(logprobs))
            assert torch.all(logprobs > -1e6)

    def test__gflownet_minimal_runs(self, n_repeat=1, batch_size=2):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "batch_size") and method_name in self.batch_size:
            batch_size = self.batch_size[method_name]

        for _ in range(n_repeat):
            # Load config by setting batch size of 2 and 1 train step.
            config = load_base_test_config(
                overrides=[
                    f"gflownet.optimizer.batch_size.forward={batch_size}",
                    "gflownet.optimizer.n_train_steps=1",
                ]
            )

            # Initialize a GFlowNet agent from the configuration file
            gflownet = gflownet_from_config(config, env=self.env)

            # Train
            gflownet.train()
            assert True


class BaseTestsDiscrete(BaseTestsCommon):
    def test__init__state_is_source_no_parents(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            assert self.env.is_source(self.env.state)
            parents, actions = self.env.get_parents()
            assert len(parents) == 0
            assert len(actions) == 0

    def test__reset__state_is_source_no_parents(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.step_random()
            self.env.reset()
            assert self.env.is_source(self.env.state)
            parents, actions = self.env.get_parents()
            assert len(parents) == 0
            assert len(actions) == 0

    def test__get_parents_step_get_mask__are_compatible(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            n_actions = 0
            while not self.env.done:
                state = copy(self.env.state)
                # Sample random action
                state_next, action, valid = self.env.step_random()
                assert valid
                n_actions += 1
                assert n_actions <= self.env.max_traj_length
                assert self.env.n_actions == n_actions
                parents, parents_a = self.env.get_parents()
                assert any([self.env.equal(p, state) for p in parents])
                assert len(parents) == len(parents_a)
                for p, p_a in zip(parents, parents_a):
                    mask = self.env.get_mask_invalid_actions_forward(p, False)
                    assert self.env.action2representative(
                        p_a
                    ) in self.env.get_valid_actions(mask, p, False)

    def test__get_parents__all_parents_are_reached_with_different_actions(
        self, n_repeat=1
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                state = copy(self.env.state)
                # Sample random action
                state_next, action, valid = self.env.step_random()
                assert valid
                _, parents_a = self.env.get_parents()
                assert len(set(parents_a)) == len(parents_a)

    def test__state2readable__is_reversible(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                state_recovered = self.env.readable2state(self.env.state2readable())
                if state_recovered is not None:
                    assert self.env.isclose(self.env.state, state_recovered)
                self.env.step_random()

    def test__get_parents__returns_same_state_and_eos_if_done(
        self, n_repeat=1, n_states=3
    ):
        method_name = _get_current_method_name()

        if hasattr(self, "n_states") and method_name in self.n_states:
            n_states = self.n_states[method_name]

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            for state in states:
                self.env.set_state(state, done=True)
                parents, actions = self.env.get_parents()
                assert all([self.env.equal(p, self.env.state) for p in parents])
                assert actions == [self.env.eos]

    def test__actions2indices__returns_expected_tensor(self, n_repeat=1, batch_size=2):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        if hasattr(self, "batch_size") and method_name in self.batch_size:
            batch_size = self.batch_size[method_name]

        for _ in range(n_repeat):
            action_space = self.env.action_space_torch
            indices_rand = torch.randint(
                low=0,
                high=action_space.shape[0],
                size=(batch_size,),
            )
            actions = action_space[indices_rand, :]
            action_indices = self.env.actions2indices(actions)
            assert torch.equal(action_indices, indices_rand)
            assert torch.equal(action_space[action_indices], actions)


class BaseTestsContinuous(BaseTestsCommon):
    def test__reset__state_is_source(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            self.env.step_random()
            self.env.reset()
            assert self.env.is_source(self.env.state)

    def test__sampling_forwards_reaches_done_in_finite_steps(self, n_repeat=1):
        method_name = _get_current_method_name()

        if hasattr(self, "repeats") and method_name in self.repeats:
            n_repeat = self.repeats[method_name]

        for _ in range(n_repeat):
            n_actions = 0
            while not self.env.done:
                # Sample random action
                state_next, action, valid = self.env.step_random()
                n_actions += 1
                assert n_actions <= self.env.max_traj_length


def _get_current_method_name():
    """Helper which returns the name of the current method as a string."""
    return inspect.currentframe().f_back.f_code.co_name


def _get_terminating_states(env, n):
    if hasattr(env, "get_all_terminating_states"):
        return env.get_all_terminating_states()
    elif hasattr(env, "get_grid_terminating_states"):
        return env.get_grid_terminating_states(n)
    elif hasattr(env, "get_uniform_terminating_states"):
        return env.get_uniform_terminating_states(n, 0)
    elif hasattr(env, "get_random_terminating_states"):
        return env.get_random_terminating_states(n, 0)
    else:
        warnings.warn(
            f"""
        Testing backward sampling or setting terminating states requires that
        the environment implements one of the following:
            - get_all_terminating_states()
            - get_grid_terminating_states()
            - get_uniform_terminating_states()
            - get_random_terminating_states()
        Environment {env.__class__} does not have any of the above, therefore
        backward sampling will not be tested.
        """
        )
        return None


# TODO: None of these tests are used. Remove?
def test__state2policy__is_reversible(env):
    env = env.reset()
    while not env.done:
        state_recovered = env.policy2state(env.state2policy())
        if state_recovered is not None:
            assert env.equal(env.state, state_recovered)
        env.step_random()


def test__get_parents__returns_no_parents_in_initial_state(env):
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0


def test__default_config_equals_default_args(env, env_config_path):
    with open(env_config_path, "r") as f:
        config_env = yaml.safe_load(f)
    config_env = hydra.utils.instantiate(config_env)
    assert True
