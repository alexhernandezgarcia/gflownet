import pytest
import torch

from gflownet.envs.scrabble import Scrabble
from gflownet.proxy.scrabble import ScrabbleScorer


@pytest.fixture()
def proxy():
    return ScrabbleScorer(vocabulary_check=True, device="cpu", float_precision=32)


@pytest.fixture
def env():
    return Scrabble(max_length=7, device="cpu")


@pytest.mark.parametrize(
    "samples, scores_expected",
    [
        (
            [
                ["C", "A", "T", "0", "0", "0", "0"],
                ["D", "O", "G", "0", "0", "0", "0"],
                ["B", "I", "R", "D", "0", "0", "0"],
                ["F", "R", "I", "E", "N", "D", "S"],
            ],
            [3 + 1 + 1, 2 + 1 + 2, 3 + 1 + 1 + 2, 4 + 1 + 1 + 1 + 1 + 2 + 1],
        ),
    ],
)
def test__scrabble_scorer__returns_expected_scores_list_input_list_tokens(
    env, proxy, samples, scores_expected
):
    proxy.setup(env)
    scores = proxy(samples)
    assert scores.tolist() == scores_expected


@pytest.mark.parametrize(
    "samples, scores_expected",
    [
        (
            ["CAT", "DOG", "BIRD", "FRIENDS"],
            [3 + 1 + 1, 2 + 1 + 2, 3 + 1 + 1 + 2, 4 + 1 + 1 + 1 + 1 + 2 + 1],
        ),
    ],
)
def test__scrabble_scorer__returns_expected_scores_input_list_strings(
    env, proxy, samples, scores_expected
):
    proxy.setup(env)
    scores = proxy(samples)
    assert scores.tolist() == scores_expected


@pytest.mark.parametrize(
    "sample, score_expected",
    [
        (
            "C A T",
            3 + 1 + 1,
        ),
        (
            "C A T Z",
            0,
        ),
        (
            "D O G",
            2 + 1 + 2,
        ),
        (
            "B I R D",
            3 + 1 + 1 + 2,
        ),
        (
            "G F N",
            0,
        ),
        (
            "F R I E N D S",
            4 + 1 + 1 + 1 + 1 + 2 + 1,
        ),
    ],
)
def test__scrabble_scorer__returns_expected_scores_input_state2proxy(
    env, proxy, sample, score_expected
):
    proxy.setup(env)
    env.set_state(env.readable2state(sample))
    sample_proxy = env.state2proxy()
    score = proxy(sample_proxy)
    assert score.tolist() == [score_expected]
