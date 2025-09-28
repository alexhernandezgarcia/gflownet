import pytest
import torch

from gflownet.envs.crystals.crystal import Crystal
from gflownet.proxy.crystals.corners import CrystalCorners


@pytest.fixture()
def proxy_default():
    return CrystalCorners(device="cpu", float_precision=32)


@pytest.fixture()
def config_one_sg():
    return [{"spacegroup": 225, "mu": 0.85, "sigma": 0.05}]


@pytest.fixture()
def config_two_sg():
    return [
        {"spacegroup": 225, "mu": 0.85, "sigma": 0.05},
        {"spacegroup": 229, "mu": 0.65, "sigma": 0.05},
    ]


@pytest.fixture()
def config_one_el():
    return [{"element": 46, "mu": 0.75, "sigma": 0.1}]


@pytest.fixture()
def config_two_el():
    return [
        {"element": 46, "mu": 0.75, "sigma": 0.1},
        {"element": 78, "mu": 0.75, "sigma": 0.001},
    ]


@pytest.fixture()
def config_all():
    return [
        {"element": 46, "mu": 0.75, "sigma": 0.1},
        {"element": 78, "mu": 0.75, "sigma": 0.001},
        {"spacegroup": 225, "mu": 0.85, "sigma": 0.05},
        {"spacegroup": 229, "mu": 0.65, "sigma": 0.05},
    ]


@pytest.fixture()
def config_invalid_both_spacegroup_element():
    return [
        {"spacegroup": 225, "element": 46, "mu": 0.75, "sigma": 0.1},
        {"spacegroup": 225, "mu": 0.85, "sigma": 0.05},
        {"spacegroup": 229, "mu": 0.65, "sigma": 0.05},
    ]


@pytest.fixture()
def config_invalid_no_spacegroup_or_element():
    return [
        {"mu": 0.75, "sigma": 0.1},
        {"spacegroup": 225, "mu": 0.85, "sigma": 0.05},
        {"spacegroup": 229, "mu": 0.65, "sigma": 0.05},
    ]


@pytest.fixture()
def config_invalid_mu_missing():
    return [
        {"spacegroup": 225, "sigma": 0.05},
        {"spacegroup": 229, "mu": 0.65, "sigma": 0.05},
    ]


@pytest.fixture()
def config_invalid_sigma_missing():
    return [
        {"spacegroup": 225, "sigma": 0.05},
        {"spacegroup": 229, "mu": 0.65, "sigma": 0.05},
    ]


@pytest.fixture
def env_gull():
    return Crystal(
        composition_kwargs={
            "elements": [78, 46],
            "min_diff_elem": 1,
            "max_diff_elem": 1,
            "min_atoms": 2,
            "max_atoms": 4,
            "min_atom_i": 2,
            "max_atom_i": 4,
            "do_charge_check": False,
        },
        space_group_kwargs={"space_groups_subset": [225, 229]},
        lattice_parameters_kwargs={
            "min_length": 2.0,
            "max_length": 4.0,
            "min_angle": 60.0,
            "max_angle": 140.0,
        },
        do_sg_to_composition_constraints=True,
        do_sg_before_composition=True,
    )


def test__proxy_default__initializes_properly(proxy_default):
    assert True


@pytest.mark.parametrize(
    "config",
    [
        "config_one_sg",
        "config_two_sg",
        "config_one_el",
        "config_two_el",
        "config_all",
        "config_invalid_both_spacegroup_element",
        "config_invalid_no_spacegroup_or_element",
        "config_invalid_mu_missing",
        "config_invalid_sigma_missing",
    ],
)
def test__dict_is_valid(config, request):
    is_valid = "invalid" not in config
    config = request.getfixturevalue(config)
    assert all([CrystalCorners._dict_is_valid(el) for el in config]) == is_valid


def test__setup__works_as_expected(proxy_default, env_gull):
    proxy_default.setup(env_gull)
    assert hasattr(proxy_default, "min_length")
    assert hasattr(proxy_default, "max_length")
    assert proxy_default.min_length == 2.0
    assert proxy_default.max_length == 4.0


@pytest.mark.parametrize(
    "lp_lengths, lp_lengths_corners",
    [
        (
            torch.tensor(
                [
                    [2.0, 2.0, 2.0],
                    [4.0, 4.0, 4.0],
                    [2.0, 3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        ),
        (
            torch.tensor(
                [
                    [2.0, 2.5, 3.0, 3.5, 4.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [-1.0, -0.5, 0.0, 0.5, 1.0],
                ],
                dtype=torch.float32,
            ),
        ),
    ],
)
def test__lattice_lengths_to_corners_proxy__returns_expected(
    proxy_default, env_gull, lp_lengths, lp_lengths_corners
):
    proxy_default.setup(env_gull)
    assert torch.equal(
        proxy_default.lattice_lengths_to_corners_proxy(lp_lengths), lp_lengths_corners
    )


def test__proxy_default__returns_valid_scores(proxy_default, env_gull, n_states=10):
    proxy_default.setup(env_gull)

    # Generate random crystal terminating states
    states = env_gull.get_random_terminating_states(n_states)
    states_proxy = env_gull.states2proxy(states)

    # Compute scores
    scores = proxy_default(states_proxy)

    assert not any(torch.isnan(scores))
    assert torch.all(scores > 0)


@pytest.mark.parametrize(
    "config",
    [
        "config_one_sg",
        "config_two_sg",
        "config_one_el",
        "config_two_el",
        "config_all",
    ],
)
def test__proxies_of_conditions_return_expected_values(config, env_gull, request):
    config = request.getfixturevalue(config)
    # Initialize proxy with condition
    proxy = CrystalCorners(device="cpu", float_precision=32, config=config)
    proxy.setup(env_gull)

    assert config == proxy.proxies

    # Check that mu is a maximum
    for c in proxy.proxies:
        proxy_mu = c["proxy"](torch.tensor(c["mu"]).repeat(1, 3))[0]
        assert proxy_mu > c["proxy"](torch.tensor(c["mu"] + 0.01).repeat(1, 3))[0]
        assert proxy_mu > c["proxy"](torch.tensor(c["mu"] - 0.01).repeat(1, 3))[0]


def test__applying_sg_condition_changes_scores(
    proxy_default, config_one_sg, env_gull, n_states=10
):
    # Set up default proxy
    proxy_default.setup(env_gull)

    # Initialize proxy with condition
    proxy_cond = CrystalCorners(device="cpu", float_precision=32, config=config_one_sg)
    proxy_cond.setup(env_gull)

    # Generate random crystal terminating states
    states = env_gull.get_random_terminating_states(n_states)
    states_proxy = env_gull.states2proxy(states)

    # Compute scores with default proxy
    scores_default = proxy_default(states_proxy)
    # Compute scores with proxy with condition
    scores_cond = proxy_cond(states_proxy)

    # Check that scores are different if space group matches condition
    for sg, score_default, score_cond in zip(
        states_proxy[:, -7], scores_default, scores_cond
    ):
        if sg == 225:
            assert score_default != score_cond
        else:
            assert score_default == score_cond


def test__applying_el_condition_changes_scores(
    proxy_default, config_one_el, env_gull, n_states=10
):
    # Set up default proxy
    proxy_default.setup(env_gull)

    # Initialize proxy with condition
    proxy_cond = CrystalCorners(device="cpu", float_precision=32, config=config_one_el)
    proxy_cond.setup(env_gull)

    # Generate random crystal terminating states
    states = env_gull.get_random_terminating_states(n_states)
    states_proxy = env_gull.states2proxy(states)

    # Compute scores with default proxy
    scores_default = proxy_default(states_proxy)
    # Compute scores with proxy with condition
    scores_cond = proxy_cond(states_proxy)

    # Check that scores are different if element matches condition
    for comp, score_default, score_cond in zip(
        states_proxy[:, :-7], scores_default, scores_cond
    ):
        if comp[46] > 0:
            assert score_default != score_cond
        else:
            assert score_default == score_cond
