import pytest

from gflownet.envs.toy import Toy
from gflownet.proxy.toy import ToyScorer


@pytest.fixture()
def proxy():
    return ToyScorer()


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
            [30, 14, 23, 10, 30, 5],
        ),
        (
            [[3], [0], [3], [1], [3], [6], [8], [10], [0]],
            [30, 0, 30, 0, 30, 23, 10, 5, 0],
        ),
    ],
)
def test__toy_scorer__returns_expected_scores(env, proxy, samples, scores_expected):
    proxy.setup(env)
    scores = proxy(env.states2proxy(samples))
    assert scores.tolist() == scores_expected
