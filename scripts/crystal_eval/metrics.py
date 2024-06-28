import itertools
import multiprocessing
import os
from functools import partial
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import smact
import tqdm
from mp_api.client.mprester import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact import Element
from smact.data_loader import lookup_element_oxidation_states_custom as oxi_custom
from smact.screening import pauling_test, smact_filter
from tqdm import tqdm

from gflownet.utils.crystals import pyxtal_cache

PhaseDiagram.numerical_tol = 1e-1  # set to avoid errors with Mg, should not effect results except Mg formation energy.
LAT_TOL = 0.1


class BaseMetric:
    def __init__(self) -> None:
        self.__name__ = self.__class__.__name__

    def compute(self, structures: Iterable[Structure], **kwargs) -> dict:
        """Performs the computational part of a metric and returns a JSONable dictionnary

        Parameters
        ----------
        structures : Structure
        compositions : Composition

        Returns
        -------
        dict
            jsonable dictionary
        """
        return {}

    def plot(self, data_results: dict[str]):
        """Create various plots and prints given data_results dict. The data_result dic has as
        keys different datasets names and as values a dictionnary providing from the above
        compute function. You can typically print things or save plots to out_dir.

        Parameters
        ----------
        data_results :  dict[str]
            keys different datasets names and as values a dictionnary providing from the above
        compute function.

        Returns
        -------
        None
        """
        return None


class NumberOfElements(BaseMetric):
    def compute(self, structures: Iterable[Structure], **kwargs) -> dict:
        n_elems = [len(s.composition.as_dict()) for s in structures]
        n_elems_distr = pd.Series(n_elems).value_counts().to_dict()
        return n_elems_distr

    def plot(self, data_results: dict[str]):
        for dataset_name, data in data_results.items():
            print(f"Dataset {dataset_name}")
            for n, amount in data.items():
                print(f"Number of {n}-elements materials:{amount}")
            print("")

        df = pd.DataFrame()
        for label, data in data_results.items():
            temp_df = pd.DataFrame(
                {
                    "Elements": list(data.keys()),
                    "Occurrences": list(data.values()),
                    "Dataset": label,
                }
            )
            df = pd.concat([df, temp_df], ignore_index=True)

        # Create the plot using Seaborn
        fig, ax = plt.subplots()
        sns.barplot(x="Elements", y="Occurrences", hue="Dataset", data=df, ax=ax)

        # Labeling the plot
        ax.set_xlabel("Elements")
        ax.set_ylabel("Occurrences")
        ax.set_title("Comparison of Element Occurrences")
        fig.savefig("number_of_elements.pdf")


class Rediscovery(BaseMetric):
    def __init__(self, rediscovery_path=None):
        super().__init__()
        if rediscovery_path is not None:
            self.ref = pd.read_csv(rediscovery_path)
        else:
            try:
                key = os.environ.get("MATPROJ_API_KEY")
                if key is None:
                    print(
                        "No MP Key. Set your env variable MATPROJ_API_KEY or remove the Rediscovery metric."
                    )
                    exit()
                self.ref = MPRester(key)
            except (KeyError, ValueError):
                print(
                    "No reference (either dataset or Materials Project API Key) present."
                )
                exit()

    def compute(self, structures, **kwargs):
        compositions = [s.composition.as_dict() for s in structures]
        matches = self._comp_rediscovery(compositions, self.ref)
        return {"matches": matches}

    def plot(self, data_results, out_dir):
        for data_name, results in data_results.items():
            print(f"Following matches were found for {data_name}")
            print(results)
            print(results["matches"])

    def _check_ref(self, query, ref):
        print(ref)
        if isinstance(ref, pd.DataFrame):
            ref = ref[ref.columns[8:-2]]
            for col in ref.columns:
                if col not in query:
                    query[col] = 0
            query_df = pd.Series(query)
            found = ref.loc[ref.eq(query_df).all(axis=1)]
            if len(found) > 0:
                return query, found.to_dict("index")
        elif isinstance(ref, MPRester):
            query_crit = [k for k, v in query.items() if v > 0]
            comp = "-".join(query_crit)
            docs = ref.get_structures(comp)
            for doc in docs:
                # for the entries returned, get the conventional structure
                # unreduced composition dictionary
                struc = doc
                SGA = SpacegroupAnalyzer(struc)
                struc = SGA.get_conventional_standard_structure()

                doc_comp = dict(struc.composition.get_el_amt_dict())
                if comp == doc_comp:
                    return query, doc_comp
        else:
            raise TypeError("Query cannot be made against reference")

        return (None, None)

    def _comp_rediscovery(self, compositions, reference):
        match_dix = {}
        for i, c in enumerate(tqdm(compositions)):
            comp_dic = c
            k, v = self._check_ref(comp_dic, reference)
            if v:
                match_dix[k] = v
        return match_dix


class SMACT(BaseMetric):
    def __init__(self, oxidation_states_set="default", oxidation_only=False) -> None:
        super().__init__()
        self.oxidation_states_set = oxidation_states_set
        self.oxidation_only = oxidation_only
        if oxidation_only:
            self.__name__ = "SMACTNeutral-" + oxidation_states_set
        else:
            self.__name__ = self.__class__.__name__ + "-" + oxidation_states_set

    def compute(self, structures, **kwargs):
        res = {"formulae": [], "passes": []}
        for s in structures:
            comp = s.composition
            comp_dix = comp.as_dict()
            res["formulae"].append(comp.formula)
            res["passes"].append(
                int(
                    smact_validity(
                        comp_dix,
                        oxidation_states_set=self.oxidation_states_set,
                        use_pauling_test=not self.oxidation_only,
                    )
                )
            )
        return res

    def plot(self, data_results):
        fig, ax = plt.subplots(1, 1)
        summ_dict = {"Data": [], "Pass Count": [], "Success rate": []}
        for label, results in data_results.items():
            passes = sum(results["passes"])
            total = len(results["passes"])
            summ_dict["Data"].append(label)
            summ_dict["Pass Count"].append(passes)
            summ_dict["Success rate"].append(passes * 100 / total)
            print(
                f"{self.__name__} succes rate for {label}: {passes * 100 / total:.1f}%"
            )
        # df = pd.DataFrame(summ_dict)
        # sns.barplot(data=df, x="Data", y="Pass Count", ax=ax, hue="Data")
        # labels = [f"{i:.1f}%" for i in summ_dict["Success rate"]]
        # ax.bar_label(ax.containers[-1], labels=labels)
        # ax.set_title("SMACT: Charge Neutrality + Electronegativity")
        # ax.set_xlabel("Data")
        # ax.set_ylabel("Counts")
        fig.savefig("smact_test.png")


def smact_validity(
    comp_dict,
    use_pauling_test=True,
    include_alloys=True,
    oxidation_states_set="default",
):
    elem_symbols = tuple(comp_dict.keys())
    count = [int(c) for c in comp_dict.values()]
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]

    # Select the specified oxidation states set:
    oxi_set = {
        "default": [e.oxidation_states for e in smact_elems],
        "icsd": [e.oxidation_states_icsd for e in smact_elems],
        "pymatgen": [e.oxidation_states_sp for e in smact_elems],
        "wiki": [e.oxidation_states_wiki for e in smact_elems],
    }
    if oxidation_states_set in oxi_set:
        ox_combos = oxi_set[oxidation_states_set]
    elif os.path.exists(oxidation_states_set):
        ox_combos = [oxi_custom(e.symbol, oxidation_states_set) for e in smact_elems]

    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


class Eform(BaseMetric):
    """Metric that processes the formation energy

    Following are printed or plotted
        * % materials with negative energy
        * top-k materials
        * Violin plot of distribution
    """

    def __init__(self, k=10) -> None:
        super().__init__()
        self.k = k

    def compute(
        self, structures: Iterable[Structure], energies: list, **kwargs
    ) -> dict:
        energies = np.array(energies)
        total_am = len(energies)
        neg_am = (energies < 0).sum()
        neg_frac = neg_am / total_am * 100

        topk_idx = np.argsort(energies)[: self.k]
        topk = [structures[idx].formula for idx in topk_idx]

        return {"neg_frac": neg_frac, "topk": topk, "energies": list(energies)}

    def plot(self, data_results: dict[str]):
        energy_dic = {"dataset": [], "energy": []}
        for dataset_name, data in data_results.items():
            print(f"Dataset {dataset_name}")
            neg_frac, topk, energies = data["neg_frac"], data["topk"], data["energies"]

            print(f"Fraction of samples that have a negative eform: {neg_frac:.1f}%")
            print(f"Top {len(topk)} materials: {topk}")

            energy_dic["dataset"].extend([dataset_name] * len(energies))
            energy_dic["energy"].extend(energies)
        df = pd.DataFrame(energy_dic)
        fig, ax = plt.subplots()
        sns.violinplot(x="dataset", y="energy", data=df, ax=ax)
        fig.savefig("eform_distribution.pdf")
        return None


def _init_worker(hull_path):
    global MP_HULL
    MP_HULL = pd.read_pickle(hull_path)


def _ehull_worker(composition):
    ehull = MP_HULL.get_hull_energy_per_atom(Composition(composition))
    return composition, ehull


class Ehull(BaseMetric):
    """Metric that computes the energy above hull
    It requires a path to a fitted and pickled PhaseDiagram,
    computed on the required reference, e.g. Materials Project.
    This has already been computed on the Materials Project, ask for it, or/and use the path to it on the cluster.

    Following are printed or plotted
        * % materials below 0.25 eV/atom
        * top-k materials
        * Violin plot of distribution
    """

    def __init__(
        self, PD_path, k=10, stab_thresh=0, n_jobs=1, debug=False, element_set=None
    ) -> None:
        super().__init__()
        self.k = k
        self.PD_path = PD_path
        self.stab_thresh = stab_thresh
        self.debug = debug
        self.n_jobs = n_jobs
        if element_set is None:
            self.element_set = set(
                ["H", "Li", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Fe"]
            )

    def compute(
        self, structures: Iterable[Structure], energies: list, **kwargs
    ) -> dict:

        all_compositions = [s.composition for s in structures]
        all_energies = energies
        compositions = []
        energies = []
        for comp, energy in zip(all_compositions, all_energies):
            drop = False
            for element in comp.as_dict().keys():
                if element not in self.element_set:
                    drop = True
            if not drop:
                compositions.append(str(comp))
                energies.append(energy)
        if self.debug:
            print("Debug is true, computing a subset.")
            compositions = compositions[:5]
            energies = energies[:5]

        # set pool of workers
        init_worker_with_path = partial(_init_worker, self.PD_path)
        if self.n_jobs > 1:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ctx.Pool(
                processes=self.n_jobs, initializer=init_worker_with_path
            )
        else:
            self.pool = None
            init_worker_with_path()

        # perform ehull computations
        if self.pool is not None:
            results = {}
            for composition, ehull in tqdm(
                self.pool.imap_unordered(_ehull_worker, compositions, chunksize=1),
                total=len(compositions),
            ):
                results[composition] = ehull
            self.pool.close()
            ehulls = [results[composition] for composition in compositions]
        else:
            ehulls = list(
                tqdm(map(_ehull_worker, compositions), total=len(compositions))
            )
            ehulls = [eh[1] for eh in ehulls]

        e_above_hull = np.array(energies) - np.array(ehulls)

        return {
            "formulas": compositions,
            "ehulls": list(e_above_hull),
        }

    def plot(self, data_results: dict[str]):
        energy_dic = {"dataset": [], "ehulls": []}
        for dataset_name, data in data_results.items():
            print(f"Dataset {dataset_name}")
            ehulls, formulas = data["ehulls"], data["formulas"]

            total_am = len(ehulls)
            neg_am = (np.array(ehulls) < self.stab_thresh).sum()
            neg_frac = neg_am / total_am * 100

            topk_idx = np.argsort(ehulls)[: self.k]
            topk = [formulas[idx] for idx in topk_idx]

            print(
                f"Fraction of samples that have ehull below {self.stab_thresh} eV/atom: {neg_frac:.1f}%"
            )
            print(f"Top {len(topk)} materials: {topk}")

            energy_dic["dataset"].extend([dataset_name] * len(ehulls))
            energy_dic["ehulls"].extend(ehulls)
        df = pd.DataFrame(energy_dic)
        fig, ax = plt.subplots()
        sns.violinplot(x="dataset", y="ehulls", data=df, ax=ax)
        ax.set_ylabel("Energy above hull [eV/atom]")
        fig.savefig("ehull_distribution.pdf")
        return None


class Comp2SG(BaseMetric):
    def compute(self, structures, sg, **kwargs):
        res = {"formulae": [], "passes": []}
        for i, s in enumerate(structures):
            comp = s.composition
            elem_nums = [v for v in comp.as_dict().values()]
            test = pyxtal_cache.space_group_check_compatible(
                int(sg[i]["spacegroup"]), elem_nums
            )
            res["formulae"].append(comp.formula)
            res["passes"].append(int(test))
        return res

    def plot(self, data_results):
        fig, ax = plt.subplots(1, 1)
        summ_dict = {"Data": [], "Pass Count": [], "Success rate": []}
        for label, results in data_results.items():
            passes = sum(results["passes"])
            total = len(results["passes"])
            summ_dict["Data"].append(label)
            summ_dict["Pass Count"].append(passes)
            summ_dict["Success rate"].append(passes * 100 / total)
        df = pd.DataFrame(summ_dict)
        sns.barplot(data=df, x="Data", y="Success rate", ax=ax, hue="Data")
        labels = [f"{i:.1f}%" for i in summ_dict["Success rate"]]
        # print(ax.containers)
        # for i, l in enumerate(labels):
        #     ax.bar_label(ax.containers[i], labels=l)
        ax.bar_label(ax.containers[-1], labels=labels)
        ax.set_title("Compatibilty Check b/w Composition and Space Group")
        ax.set_xlabel("Data")
        ax.set_ylabel("Success %")
        fig.savefig("comp2sg_test.pdf")


class SG2LP(BaseMetric):
    def compute(self, structures, sg, **kwargs):
        res = {"formulae": [], "passes": []}
        for i, s in enumerate(structures):
            comp = s.composition
            lat_sys = self._true_system(int(sg[i]["spacegroup"]))
            lattice = s.lattice
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles
            test = self._lat_sys_check(a, b, c, alpha, beta, gamma, lat_sys)
            if lat_sys != test:
                print(lat_sys, test)
                if test != "Incorrect":
                    print(a, b, c)
                    print(alpha, beta, gamma)
            res["formulae"].append(comp.formula)
            res["passes"].append(int(test == lat_sys))
        return res

    def plot(self, data_results):
        fig, ax = plt.subplots(1, 1)
        summ_dict = {"Data": [], "Pass Count": [], "Success rate": []}
        for label, results in data_results.items():
            passes = sum(results["passes"])
            total = len(results["passes"])
            summ_dict["Data"].append(label)
            summ_dict["Pass Count"].append(passes)
            summ_dict["Success rate"].append(passes * 100 / total)
        df = pd.DataFrame(summ_dict)
        sns.barplot(data=df, x="Data", y="Success rate", ax=ax, hue="Data")
        labels = [f"{i:.1f}%" for i in summ_dict["Success rate"]]
        ax.bar_label(ax.containers[0], labels=labels)
        ax.set_title("Compatibilty Check for lattice systems and lattice params")
        ax.set_xlabel("Data")
        ax.set_ylabel("Success %")
        fig.savefig("sg2lp_test.pdf")

    def _lat_sys_check(self, a, b, c, alpha, beta, gamma, true_sys):
        all_sys = [
            "cubic",
            "hexagonal",
            "monoclinic",
            "orthorhombic",
            "rhombohedral",
            "tetragonal",
            "triclinic",
        ]
        abc_opt = [1 for _ in all_sys]

        if a > b - LAT_TOL and a < b + LAT_TOL:
            if b > c - LAT_TOL and b < c + LAT_TOL:
                not_abc = [1, 2, 3, 5, 6]
            else:
                not_abc = [0, 2, 3, 4, 6]
        else:
            not_abc = [0, 1, 4, 5]
        for i in not_abc:
            abc_opt[i] = 0
        angles_opt = [1 for _ in all_sys]
        not_ang = []

        if alpha > beta - LAT_TOL and alpha < beta + LAT_TOL:
            if beta > gamma - LAT_TOL and beta < gamma + LAT_TOL:
                if beta > 90 - LAT_TOL and beta < 90 + LAT_TOL:
                    not_ang = [1, 2, 4, 6]
                else:
                    not_ang = [0, 1, 2, 3, 5, 6]
            elif gamma > 120 - LAT_TOL and gamma < 120 + LAT_TOL:
                if beta > 90 - LAT_TOL and beta < 90 + LAT_TOL:
                    not_ang = [0, 2, 3, 4, 5, 6]
        elif alpha > gamma - LAT_TOL and alpha < gamma + LAT_TOL:
            if gamma > 90 - LAT_TOL and gamma < 90 + LAT_TOL:
                not_ang = [0, 1, 3, 4, 5, 6]
        else:
            not_ang = [0, 1, 2, 3, 4, 5]
        for i in not_ang:
            angles_opt[i] = 0

        match = [i == 1 and j == 1 for i, j in zip(abc_opt, angles_opt)]
        only1 = sum(match)
        if only1 != 1:
            print("Fails")
            print(a, b, c)
            print(alpha, beta, gamma)
            print(abc_opt)
            print(angles_opt)
            return "Incorrect"
            # raise ValueError(f"Invalid system was selected: {only1}")

        return all_sys[match.index(1)]

    def _true_system(self, sg):
        """
        Adapted from pymatgen
        """
        if not (sg == int(sg) and 0 < sg < 231):
            raise ValueError(f"Received invalid space group {sg}")

        ls = "cubic"

        if 0 < sg < 3:
            ls = "triclinic"
        elif sg < 16:
            ls = "monoclinic"
        elif sg < 75:
            ls = "orthorhombic"
        elif sg < 143:
            ls = "tetragonal"
        elif sg < 168:
            ls = "trigonal"
        elif sg < 195:
            ls = "hexagonal"

        if sg in [146, 148, 155, 160, 161, 166, 167]:
            ls = "rhombohedral"
        if ls == "trigonal":
            ls = "hexagonal"

        return ls
