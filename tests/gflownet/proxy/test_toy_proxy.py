import pytest

from gflownet.envs.toy import Toy
from gflownet.proxy.toy import ToyScorer


@pytest.fixture()
def proxy():
    return ToyScorer()


@pytest.fixture()
def proxy_custom_values():
    return ToyScorer(values={3: 3, 4: 4, 6: 3, 8: 0, 9: 3, 10: 5})


@pytest.fixture
def env():
    return Toy()


def test__toy_scorer__initializes_properly():
    proxy = ToyScorer()
    assert proxy is not None
    assert True


@pytest.mark.parametrize(
    "samples, scores_expected",
    [
        (
            [[3], [4], [6], [8], [9], [10]],
            [30.0, 14.0, 23.0, 10.0, 30.0, 5.0],
        ),
        (
            [[3], [0], [3], [1], [3], [6], [8], [10], [0]],
            [30.0, 0.0, 30.0, 0.0, 30.0, 23.0, 10.0, 5.0, 0.0],
        ),
    ],
)
def test__toy_scorer__returns_expected_scores(env, proxy, samples, scores_expected):
    proxy.setup(env)
    scores = proxy(env.states2proxy(samples))
    assert scores.tolist() == scores_expected


@pytest.mark.parametrize(
    "samples, scores_expected",
    [
        (
            [[3], [4], [6], [8], [9], [10]],
            [3.0, 4.0, 3.0, 0.0, 3.0, 5.0],
        ),
        (
            [[3], [0], [3], [1], [3], [6], [8], [10], [0]],
            [3.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 5.0, 0.0],
        ),
    ],
)
def test__toy_scorer_custom_values__returns_expected_scores(
    env, proxy_custom_values, samples, scores_expected
):
    proxy_custom_values.setup(env)
    scores = proxy_custom_values(env.states2proxy(samples))
    assert scores.tolist() == scores_expected
