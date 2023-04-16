import pickle
from gflownet.proxy.base import Proxy

PROTONS_NUMBER_COUNTS = None


def _read_protons_number_counts():
    global PROTONS_NUMBER_COUNTS
    if PROTONS_NUMBER_COUNTS is None:
        with open(Path(__file__).parent /'number_of_protons_counts.pkl', 'rb') as handle:
            PROTONS_NUMBER_COUNTS = pickle.load(handle)
    return PROTONS_NUMBER_COUNTS


class Composition(Proxy):
    def __init__(self, normalise: bool = True, **kwargs):
        super().__init__(**kwargs)
        counts = _read_protons_number_counts()
        # next steps: 
        # - look into the env, find the max number of protons
        # - tensor of charges
