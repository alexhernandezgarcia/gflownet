import common
import pytest

from gflownet.envs.iam.investment import InvestmentDiscrete


@pytest.fixture
def env():
    return InvestmentDiscrete()


def test__environment_initializes_properly():
    env = InvestmentDiscrete()
    assert True


def test__get_right_mask(env):#check that if tech has been assigned, but sector is not, there's only one available action
    actions = [False] + [True] * (env.action_space_dim - 1)
    assert (
            env.get_mask_invalid_actions_forward(
                {"SECTOR": 0, "TAG": 0, "TECH": 1, "AMOUNT": 1}
            )
            == actions
    )
    assert (
            env.get_mask_invalid_actions_forward(
                {"SECTOR": 0, "TAG": 0, "TECH": 1, "AMOUNT": 0}
            )
            == actions
    )
    assert (
            env.get_mask_invalid_actions_forward(
                {"SECTOR": 0, "TAG": 1, "TECH": 1, "AMOUNT": 0}
            )
            == actions
    )


@pytest.mark.parametrize(
    "action_space",
    [
        [
            # SECTOR choices
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            # TAG choices
            (2, 1),
            (2, 2),
            (2, 3),
            # TECH choices
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            # AMOUNT choices
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            # EOS
            (-1, -1),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert (
            len(env.action_space)
            == len(env.sectors) + len(env.tags) + len(env.techs) + len(env.amounts) + 1
    )
    assert env.eos in env.action_space
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        # Source state has no parents
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                [],
                [],
        ),
        # State with only SECTOR assigned
        (
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # POWER sector
                [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}],
                [(1, 1)],  # SECTOR=POWER action
        ),
        # State with TECH and AMOUNT, can only be tech, else it would immediately assign sector and tag
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 1},  # power_NUCLEAR, HIGH
                [
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 1},
                ],
                [(3, 3)],
        ),
        # State with SECTOR, TECH, and AMOUNT (cannot assign amount before filling in tech, had to be there before)
        (
                {
                    "SECTOR": 1,
                    "TAG": 0,
                    "TECH": 3,
                    "AMOUNT": 1,
                },  # POWER, power_NUCLEAR, HIGH
                [
                    {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 1},
                    {
                        "SECTOR": 0,
                        "TAG": 0,
                        "TECH": 3,
                        "AMOUNT": 1,
                    },
                ],
                [(3, 3), (1, 1)],  # TECH=power_NUCLEAR action
        ),  # State with SECTOR, TAG, and AMOUNT
        (
                {
                    "SECTOR": 1,
                    "TAG": 2,
                    "TECH": 0,
                    "AMOUNT": 1,
                },  # POWER, BROWN, HIGH
                [
                    {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 1},
                    {"SECTOR": 0, "TAG": 2, "TECH": 0, "AMOUNT": 1},
                    {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 0},
                ],
                [(2, 2), (1, 1), (4, 1)],
        ),
    ],
)
def test__get_parents__returns_expected(
        env, state, parents_expected, parents_a_expected
):
    parents, parents_a = env.get_parents(state)
    assert len(parents) == len(parents_expected)
    assert len(parents_a) == len(parents_a_expected)

    # Create sets of (parent, action) pairs for robust comparison
    actual_pairs = set(
        (tuple(sorted(p.items())), a) for p, a in zip(parents, parents_a)
    )
    expected_pairs = set(
        (tuple(sorted(p.items())), a)
        for p, a in zip(parents_expected, parents_a_expected)
    )

    assert actual_pairs == expected_pairs


@pytest.mark.parametrize(
    "state, action, next_state, valid",
    [
        # From source, assign SECTOR
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                (1, 1),  # SECTOR=POWER
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                True,
        ),
        # From source, assign TAG
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                (2, 1),  # TAG=GREEN
                {"SECTOR": 0, "TAG": 1, "TECH": 0, "AMOUNT": 0},
                True,
        ),
        # From source, assign TECH directly
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                (3, 3),  # TECH=power_NUCLEAR
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                True,
        ),
        # Assign AMOUNT from TECH -> FALSE, should fill in sector and tag first
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 7, "AMOUNT": 0},
                (4, 1),  # AMOUNT=HIGH
                {"SECTOR": 0, "TAG": 0, "TECH": 7, "AMOUNT": 0},
                False,
        ),
        # EOS on missing SECTOR AND TAG
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 1},
                (-1, -1),  # EOS
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 1},
                False,
        ),
        # EOS on well defined investment
        (
                {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                (-1, -1),  # EOS
                {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                True,
        ),
        # Invalid: EOS on incomplete investment
        (
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                (-1, -1),  # EOS
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                False,
        ),
        # Invalid: Reassigning SECTOR
        (
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                (1, 2),  # Try to change SECTOR to ENERGY
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                False,
        ),
        # Invalid: incompatible Tech: nuclear on brown
        (
                {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 0},
                (3, 3),
                {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 0},
                False,
        ),
        # Fill in SECTOR
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                (1, 1),
                {"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                True,
        ),
        # Fill in TAG before SECTOR
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                (2, 1),
                {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                False,
        ),
        # Fill in TAG after SECTOR
        (
                {"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                (2, 1),
                {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 0},
                True,
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state, valid):
    env.set_state(state)
    result_state, result_action, result_valid = env.step(action)
    assert result_state == next_state
    assert result_action == action
    assert result_valid == valid


@pytest.mark.parametrize(
    "state, readable",
    [
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                "UNASSIGNED_SECTOR | UNASSIGNED_TAG | UNASSIGNED_TECH | UNASSIGNED_AMOUNT",
        ),
        (
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                "POWER | UNASSIGNED_TAG | UNASSIGNED_TECH | UNASSIGNED_AMOUNT",
        ),
        (
                {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                "POWER | GREEN | power_NUCLEAR | HIGH",
        ),
        (
                {"SECTOR": 0, "TAG": 0, "TECH": 1, "AMOUNT": 2},
                "UNASSIGNED_SECTOR | UNASSIGNED_TAG | power_COAL_noccs | MEDIUM",
        ),
    ],
)
def test__state2readable__returns_expected(env, state, readable):
    assert env.state2readable(state) == readable


@pytest.mark.parametrize(
    "readable, state",
    [
        (
                "UNASSIGNED_SECTOR | UNASSIGNED_TAG | UNASSIGNED_TECH | UNASSIGNED_AMOUNT",
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ),
        (
                "POWER | UNASSIGNED_TAG | UNASSIGNED_TECH | UNASSIGNED_AMOUNT",
                {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ),
        (
                "POWER | GREEN | power_NUCLEAR | HIGH",
                {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
        ),
        (
                "UNASSIGNED_SECTOR | UNASSIGNED_TAG | power_COAL_noccs | MEDIUM",
                {"SECTOR": 0, "TAG": 0, "TECH": 1, "AMOUNT": 2},
        ),
    ],
)
def test__readable2state__returns_expected(env, readable, state):
    assert env.readable2state(readable) == state


@pytest.mark.parametrize(
    "state, is_well_defined",
    [
        ({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}, False),  # Empty state
        ({"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0}, False),  # Only sector
        ({"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0}, False),  # Only tech
        ({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 1}, False),  # Only amount
        ({"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 1}, False),  # Tech + Amount
        ({"SECTOR": 0, "TAG": 1, "TECH": 3, "AMOUNT": 1}, False),  # missing sector
        ({"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 1}, False),  # Missing tag
        ({"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1}, True),  # Complete investment
    ],
)
def test__well_defined_investment__returns_expected(env, state, is_well_defined):
    assert env.well_defined_investment(state) == is_well_defined


def test__get_mask_invalid_actions_forward__source_state(env):
    """Test masking at source state - should allow all initial assignments but not EOS."""
    env.reset()
    mask = env.get_mask_invalid_actions_forward()

    # EOS should be masked (invalid) at source
    assert mask[-1] == True

    # All assignment actions should be valid
    for i, (choice_idx, _) in enumerate(env.action_space[:-1]):  # Exclude EOS
        assert mask[i] == False  # False means valid action


def test__get_mask_invalid_actions_forward__well_defined_state(env):
    """Test masking at well-defined state - should only allow EOS."""
    state = {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1}  # All 4 attributes assigned
    mask = env.get_mask_invalid_actions_forward(state)

    # Only EOS should be valid (False in mask)
    assert mask[-1] == False

    # All other actions should be invalid (True in mask)
    for i in range(len(mask) - 1):
        assert mask[i] == True


def test__get_uniform_terminating_states__returns_valid_states(env):
    """Test that uniform sampling produces valid terminating states."""
    states = env.get_uniform_terminating_states(10, seed=42)

    assert len(states) == 10
    for state in states:
        assert env.well_defined_investment(state)
        assert "TECH" in env.get_assigned_attributes(state)
        assert "AMOUNT" in env.get_assigned_attributes(state)
        assert "SECTOR" in env.get_assigned_attributes(state)
        assert "TAG" in env.get_assigned_attributes(state)


def test__network_structure_consistency(env):
    """Test that the network structure mappings are consistent."""
    # Check that network structures are consistent with allowed mappings
    for sector, techs in env.network_structure["sector2tech"].items():
        assert sector in env.sectors
        for tech in techs:
            assert tech in env.techs

    for sector, tags in env.network_structure["sector2tag"].items():
        assert sector in env.sectors
        for tag in tags:
            assert tag in env.tags

    for tag, techs in env.network_structure["tag2tech"].items():
        assert tag in env.tags
        for tech in techs:
            assert tech in env.techs

    for tag, sectors in env.network_structure["tag2sector"].items():
        assert tag in env.tags
        for sector in sectors:
            assert sector in env.sectors


def test__set_available_techs__masks_unavailable_techs(env):
    """Test that setting available techs properly masks unavailable options."""
    # Limit to only nuclear and solar
    env.set_available_techs([
        env.token2idx_techs['power_NUCLEAR'],
        env.token2idx_techs['power_SOLAR']
    ])

    mask = env.get_mask_invalid_actions_forward()

    # Check that coal is masked (unavailable)
    coal_action = (env.token2idx_choices['TECH'], env.token2idx_techs['power_COAL_noccs'])
    assert mask[env.action2index(coal_action)] == True

    # Check that nuclear is not masked (available)
    nuclear_action = (env.token2idx_choices['TECH'], env.token2idx_techs['power_NUCLEAR'])
    assert mask[env.action2index(nuclear_action)] == False

    # Check that solar is not masked (available)
    solar_action = (env.token2idx_choices['TECH'], env.token2idx_techs['power_SOLAR'])
    assert mask[env.action2index(solar_action)] == False


def test__set_available_techs__masks_unavailable_sectors(env):
    """Test that sectors with no available techs are also masked."""
    # Limit to only DAC technologies
    dac_techs = [env.token2idx_techs[t] for t in env.network_structure['sector2tech']['DAC']]
    env.set_available_techs(dac_techs)

    mask = env.get_mask_invalid_actions_forward()

    # POWER sector should be masked (no available techs)
    power_action = (env.token2idx_choices['SECTOR'], env.token2idx_sectors['POWER'])
    assert mask[env.action2index(power_action)] == True

    # DAC sector should not be masked (has available techs)
    dac_action = (env.token2idx_choices['SECTOR'], env.token2idx_sectors['DAC'])
    assert mask[env.action2index(dac_action)] == False


def test__set_available_techs__masks_unavailable_tags(env):
    """Test that tags with no available techs are also masked."""
    # Limit to only BROWN technologies
    brown_techs = [env.token2idx_techs[t] for t in env.network_structure['tag2tech']['BROWN']]
    env.set_available_techs(brown_techs)

    mask = env.get_mask_invalid_actions_forward()

    # GREEN tag should be masked (no available techs)
    green_action = (env.token2idx_choices['TAG'], env.token2idx_tags['GREEN'])
    assert mask[env.action2index(green_action)] == True

    # BROWN tag should not be masked (has available techs)
    brown_action = (env.token2idx_choices['TAG'], env.token2idx_tags['BROWN'])
    assert mask[env.action2index(brown_action)] == False


def test__states2proxy__returns_correct_format(env):
    """Test that states2proxy returns the expected format."""
    states = [
        {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
        {"SECTOR": 2, "TAG": 2, "TECH": 5, "AMOUNT": 2},
    ]

    processed = env.states2proxy(states)

    # Should return list of lists with single state dict each
    assert len(processed) == 2
    assert len(processed[0]) == 1
    assert len(processed[1]) == 1
    assert processed[0][0] == states[0]
    assert processed[1][0] == states[1]


class TestClimateInvestmentCommon(common.BaseTestsDiscrete):
    """Common tests for InvestmentDiscrete."""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 100,
            "test__get_parents__all_parents_are_reached_with_different_actions": 100,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 100,
        }
        self.n_states = {}  # TODO: Populate if needed