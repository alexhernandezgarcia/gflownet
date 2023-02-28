import numpy as np

def test__get_parents_step_get_mask__are_compatible(env, n=100):
    for traj in range(n):
        env = env.reset()
        while not env.done:
            mask_invalid = env.get_mask_invalid_actions_forward()
            valid_actions = [a for a, m in zip(env.action_space, mask_invalid) if not m]
            action = tuple(np.random.permutation(valid_actions)[0])
            env.step(action)
            parents, parents_a = env.get_parents()
            assert len(parents) == len(parents_a)
            for p, p_a in zip(parents, parents_a):
                mask = env.get_mask_invalid_actions_forward(p, False)
                assert p_a in env.action_space
                assert mask[env.action_space.index(p_a)] == False


def test__get_parents__returns_no_parents_in_initial_state(env):
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0

