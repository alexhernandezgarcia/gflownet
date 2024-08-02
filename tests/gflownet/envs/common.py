"""
Tests common to all environments.

Note that copying the state is necessary in some tests to preserve specific values of
the state. This is necessary because in some environments the state is a list which is
updated when an action is applied. Therefore, if the tests needs to keep older values
of the state, for example in test__trajectories_are_reversible(), a copy is needed.
"""

import inspect
import warnings
from functools import partial

import hydra
import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize

from gflownet.utils.common import copy, tbool, tfloat
from gflownet.utils.policy import parse_policy_config


class BaseTestsCommon:
    def test__set_state__creates_new_copy_of_state(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            states = _get_terminating_states(self.env, 5)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            envs = []
            for state in states:
                for idx in range(5):
                    env_new = self.env.copy().reset(idx)
                    env_new.set_state(state, done=True)
                    envs.append(env_new)
            state_ids = [id(env.state) for env in envs]
            assert len(np.unique(state_ids)) == len(state_ids)

    def test__step__returns_same_state_action_and_invalid_if_done(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

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
        self, n_repeat=1, n_states=5
    ):
        if _get_current_method_name() in self.n_states:
            n_states = self.n_states[_get_current_method_name()]

        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

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
            actions, _ = self.env.sample_actions_batch(
                policy_outputs, masks, states, is_backward=True
            )
            assert all([action == self.env.eos for action in actions])

    def test__get_logprobs__backward__returns_zero_if_done(
        self, n_repeat=1, n_states=5
    ):
        if _get_current_method_name() in self.n_states:
            n_states = self.n_states[_get_current_method_name()]

        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

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

    def test__forward_actions_have_nonzero_backward_prob(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            policy_random = torch.unsqueeze(self.env.random_policy_output, 0)
            while not self.env.done:
                state_next, action, valid = self.env.step_random(backward=False)
                if not valid:
                    continue

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

    def test__backward_actions_have_nonzero_forward_prob(
        self, n_repeat=1, n_states=100
    ):
        if _get_current_method_name() in self.n_states:
            n_states = self.n_states[_get_current_method_name()]

        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            policy_random = torch.unsqueeze(self.env.random_policy_output, 0)
            for state in states:
                self.env.set_state(state, done=True)
                while True:
                    if self.env.equal(self.env.state, self.env.source):
                        break
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
                    # state_prev = copy(state_next)  # TODO: Not accessed. Remove?

    def test__sample_backwards_reaches_source(self, n_repeat=1, n_states=100):
        if _get_current_method_name() in self.n_states:
            n_states = self.n_states[_get_current_method_name()]

        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            for state in states:
                self.env.set_state(state, done=True)
                n_actions = 0
                while True:
                    if self.env.equal(self.env.state, self.env.source):
                        assert True
                        break
                    self.env.step_random(backward=True)
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

        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()

            # Sample random forward trajectory
            states_trajectory_fw = []
            actions_trajectory_fw = []
            while not self.env.done:
                state, action, valid = self.env.step_random(backward=False)
                if valid:
                    # Copy prevents mutation by next step.
                    states_trajectory_fw.append(copy(state))
                    actions_trajectory_fw.append(action)

            # Sample backward trajectory with actions in forward trajectory
            states_trajectory_bw = []
            actions_trajectory_bw = []
            actions_trajectory_fw_copy = actions_trajectory_fw.copy()
            while not self.env.equal(self.env.state, self.env.source) or self.env.done:
                state, action, valid = self.env.step_backwards(
                    actions_trajectory_fw_copy.pop()
                )
                if valid:
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


class BaseTestsDiscrete(BaseTestsCommon):
    def test__init__state_is_source_no_parents(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            assert self.env.equal(self.env.state, self.env.source)
            parents, actions = self.env.get_parents()
            assert len(parents) == 0
            assert len(actions) == 0

    def test__reset__state_is_source_no_parents(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.step_random()
            self.env.reset()
            assert self.env.equal(self.env.state, self.env.source)
            parents, actions = self.env.get_parents()
            assert len(parents) == 0
            assert len(actions) == 0

    def test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(
        self, n_repeat=1
    ):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                policy_outputs = torch.unsqueeze(self.env.random_policy_output, 0)
                mask_invalid = self.env.get_mask_invalid_actions_forward()
                valid_actions = [
                    a for a, m in zip(self.env.action_space, mask_invalid) if not m
                ]
                masks_invalid_torch = torch.unsqueeze(torch.BoolTensor(mask_invalid), 0)
                actions, logprobs_sab = self.env.sample_actions_batch(
                    policy_outputs,
                    masks_invalid_torch,
                    [self.env.state],
                    is_backward=False,
                )
                actions_torch = torch.tensor(actions)
                logprobs_glp = self.env.get_logprobs(
                    policy_outputs=policy_outputs,
                    actions=actions_torch,
                    mask=masks_invalid_torch,
                    states_from=None,
                    is_backward=False,
                )
                action = actions[0]
                assert self.env.action2representative(action) in valid_actions
                assert torch.equal(logprobs_sab, logprobs_glp)
                self.env.step(action)

    def test__step_random__does_not_sample_invalid_actions(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                state = copy(self.env.state)
                mask = self.env.get_mask_invalid_actions_forward()
                # Sample random action
                state_next, action, valid = self.env.step_random()
                if valid is False:
                    continue
                assert mask[self.env.action2index(action)] is False
                # If action is not EOS then state should change
                if action != self.env.eos:
                    assert not self.env.equal(state, state_next)

    def test__get_parents_step_get_mask__are_compatible(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            n_actions = 0
            while not self.env.done:
                state = copy(self.env.state)
                # Sample random action
                state_next, action, valid = self.env.step_random()
                if valid is False:
                    continue
                n_actions += 1
                assert n_actions <= self.env.max_traj_length
                assert self.env.n_actions == n_actions
                parents, parents_a = self.env.get_parents()
                if torch.is_tensor(state):
                    assert any([self.env.equal(p, state) for p in parents])
                else:
                    assert state in parents
                assert len(parents) == len(parents_a)
                for p, p_a in zip(parents, parents_a):
                    mask = self.env.get_mask_invalid_actions_forward(p, False)
                    assert p_a in self.env.action_space
                    assert mask[self.env.action_space.index(p_a)] is False

    def test__get_parents__all_parents_are_reached_with_different_actions(
        self, n_repeat=1
    ):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                # Sample random action
                state_next, action, valid = self.env.step_random()
                if valid is False:
                    continue
                _, parents_a = self.env.get_parents()
                assert len(set(parents_a)) == len(parents_a)

    def test__state2readable__is_reversible(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.reset()
            while not self.env.done:
                state_recovered = self.env.readable2state(self.env.state2readable())
                if state_recovered is not None:
                    assert self.env.isclose(self.env.state, state_recovered)
                self.env.step_random()

    def test__get_parents__returns_same_state_and_eos_if_done(
        self, n_repeat=1, n_states=10
    ):
        if _get_current_method_name() in self.n_states:
            n_states = self.n_states[_get_current_method_name()]

        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            states = _get_terminating_states(self.env, n_states)
            if states is None:
                warnings.warn("Skipping test because states are None.")
                return
            for state in states:
                self.env.set_state(state, done=True)
                parents, actions = self.env.get_parents()
                assert all([self.env.equal(p, self.env.state) for p in parents])
                assert actions == [self.env.action_space[-1]]

    def test__actions2indices__returns_expected_tensor(self, n_repeat=1):
        BATCH_SIZE = 100
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            action_space = self.env.action_space_torch
            indices_rand = torch.randint(
                low=0,
                high=action_space.shape[0],
                size=(BATCH_SIZE,),
            )
            actions = action_space[indices_rand, :]
            action_indices = self.env.actions2indices(actions)
            assert torch.equal(action_indices, indices_rand)
            assert torch.equal(action_space[action_indices], actions)

    def test__gflownet_minimal_runs(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            # Load config
            with initialize(
                version_base="1.1", config_path="../../../config", job_name="xxx"
            ):
                config = compose(config_name="tests")

            # Logger
            logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

            # Proxy
            proxy = hydra.utils.instantiate(
                config.proxy,
                device=config.device,
                float_precision=config.float_precision,
            )
            evaluator = hydra.utils.instantiate(config.evaluator)

            # Policy
            forward_config = parse_policy_config(config, kind="forward")
            backward_config = parse_policy_config(config, kind="backward")
            forward_policy = hydra.utils.instantiate(
                forward_config,
                env=self.env,
                device=config.device,
                float_precision=config.float_precision,
            )
            backward_policy = hydra.utils.instantiate(
                backward_config,
                env=self.env,
                device=config.device,
                float_precision=config.float_precision,
                base=forward_policy,
            )
            config.env.buffer.train = None  # No buffers
            config.env.buffer.test = None
            config.env.buffer.replay_capacity = 0  # No replay buffer
            config.gflownet.optimizer.n_train_steps = 1  # Set 1 training step

            # GFlowNet agent
            gflownet = hydra.utils.instantiate(
                config.gflownet,
                device=config.device,
                float_precision=config.float_precision,
                env_maker=partial(self.env.copy),
                proxy=proxy,
                forward_policy=forward_policy,
                backward_policy=backward_policy,
                buffer=config.env.buffer,
                logger=logger,
                evaluator=evaluator,
            )
            gflownet.train()
            assert True


class BaseTestsContinuous(BaseTestsCommon):
    def test__reset__state_is_source(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            self.env.step_random()
            self.env.reset()
            assert self.env.equal(self.env.state, self.env.source)

    def test__sampling_forwards_reaches_done_in_finite_steps(self, n_repeat=1):
        if _get_current_method_name() in self.repeats:
            n_repeat = self.repeats[_get_current_method_name()]

        for _ in range(n_repeat):
            n_actions = 0
            while not self.env.done:
                # Sample random action
                state_next, action, valid = self.env.step_random()
                n_actions += 1
                assert n_actions <= self.env.max_traj_length

    # test__gflownet_minimal_runs(env)
    # test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env)
    # test__get_parents__returns_same_state_and_eos_if_done(env)
    # test__actions2indices__returns_expected_tensor(env)
    # test__sample_actions__get_logprobs__return_valid_actions_and_logprobs(env)


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
