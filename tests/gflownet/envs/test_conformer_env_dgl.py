import pytest
import torch
import numpy as np

from gflownet.envs.conformers.conformer_env_dgl import ConformerDGLEnv
from gflownet.utils.molecule import constants
from gflownet.utils.common import tfloat

def test_simple_creation():
    env = ConformerDGLEnv(constants.ad_smiles, save_init_pos=False)
    assert env.conformer.n_rotatable_bonds == 7


@pytest.mark.parametrize(
        'ta_indices, expected_dim', [
            (
                None,
                7
            ),
            (
                [0,1],
                2
            ),
            (
                [1,3],
                2
            ),
            (
                [2,3,4,5],
                4
            ),
            (
                [5],
                1
            )
        ]
)
def test_ta_matches_n_dim(ta_indices, expected_dim):
    env = ConformerDGLEnv(constants.ad_smiles, torsion_indices=ta_indices, 
                          save_init_pos=False)
    assert env.n_dim == expected_dim 

def test_simple_sync_angles():
    env = ConformerDGLEnv(constants.ad_smiles, 
                          save_init_pos=False)
    init_tas = env.conformer.compute_rotatable_torsion_angles()
    synced_tas = env.get_conformer_synced_with_state().compute_rotatable_torsion_angles()
    expected = torch.zeros_like(init_tas)
    assert torch.all(torch.isclose(synced_tas, expected, atol=1e-6))

def is_matched(conf, state, device, float_type):
    angles = conf.compute_rotatable_torsion_angles()
    expected = tfloat(state[:-1], device=device, float_type=float_type)
    # convert to the interval [-pi, pi]
    expected[expected > torch.pi] = expected[expected > torch.pi] - 2*torch.pi 
    return torch.all(torch.isclose(angles, expected, atol=1e-4))

def test_stress_sync_angles():
    env = ConformerDGLEnv(constants.ad_smiles, 
                          save_init_pos=False)
    for idx in range(100):
        state = (np.random.rand(env.n_dim) * 2 * np.pi).tolist() + [idx]
        conf = env.get_conformer_synced_with_state(state)
        assert is_matched(conf, state, device=env.device, float_type=env.float)

def test_conformer_sharing():
    env = ConformerDGLEnv(constants.ad_smiles, 
                          save_init_pos=False)
    state_1 = (np.random.rand(env.n_dim) * 2 * np.pi).tolist() + [0]
    state_2 = (np.random.rand(env.n_dim) * 2 * np.pi).tolist() + [1]  
    conf_1 = env.get_conformer_synced_with_state(state_1)
    assert is_matched(conf_1, state_1, device=env.device, float_type=env.float)
    conf_2 = env.get_conformer_synced_with_state(state_2)
    assert is_matched(conf_2, state_2, device=env.device, float_type=env.float)
    assert is_matched(conf_1, state_1, device=env.device, float_type=env.float)


# def test_tmp():
#     env = ConformerDGLEnv(constants.ad_smiles, 
#                           save_init_pos=False)
#     import ipdb; ipdb.set_trace()
#     an = env.conformer.get_atomic_numbers()
#     assert True


