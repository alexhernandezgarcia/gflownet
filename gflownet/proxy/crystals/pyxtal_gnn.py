from importlib.metadata import PackageNotFoundError, version

import pymatgen
import pymatgen.core
import pyxtal
import torch
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)
from pyxtal.lattice import Lattice
from pyxtal.msg import Comp_CompatibilityError
from pyxtal.symmetry import Group
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

# This is to revert a non-backwards-compatible change that was made since
# the moment the ocdata package was developped. Unfortunately, we can't use
# an older version of pymatgen from before this change because such an old
# version would not be compatible with PyXtal.
pymatgen.Composition = pymatgen.core.Composition

# URLs to the code repositories used by this proxy.
DAVE_REPO_URL = "https://github.com/sh-divya/ActiveLearningMaterials.git"

def ensure_library_version(library_name, repo_url, release=None):
    """Ensure that a library is available for import with a given version

    If the library is unavailable, or the wrong release, this method will print
    instructions to remedy the situation and raise an exception.
    """

    pip_url = f"{repo_url}@{release}"

    try:
        lib_version = version(library_name)
    except PackageNotFoundError:
        print(f"  💥 `{library_name}` cannot be imported.")
        print("    Install with:")
        print(f"    $ pip install git+{pip_url}\n")

        raise PackageNotFoundError(f"Library `{library_name}` not found")


def is_valid_crystal(
    space_group, elements, num_ions, a, b, c, alpha, beta, gamma
) -> bool:
    """Determines if a crystal is valid

    This function may not catch all instances of invalid crystals so return value
    of True should not be taken as a garantee of validity but, rather, as an
    indication that this limited function didn't find an issue with the crystal
    """
    # Validate that lengths are strictly positive
    for length in [a, b, c]:
        if length <= 0:
            return False

    # Validate that angles are individually reasonnable
    for angle in [alpha, beta, gamma]:
        if angle <= 0 or angle >= 180:
            return False

    # Validate that the angles can be use together to create a unit cell with
    # non-zero, non-imaginary volume.
    if alpha + beta + gamma >= 360:
        return False
    elif alpha + beta - gamma <= 0:
        return False
    elif alpha + gamma - beta <= 0:
        return False
    elif beta + gamma + alpha <= 0:
        return False

    return True


class PyxtalGNN(Proxy):
    """
    Proxy using a GNN model to evaluate samples.

    This proxy assumes that the samples do not contain atom coordinates and
    therefore uses PyXtal to sample atom coordinates before feeding the samples
    to the GNN model.
    
    Requirements :
    - OCD github repo cloned locally accessible from the pythonpath
    """

    ENERGY_INVALID_SAMPLE = 10

    def __init__(self, ckpt_path=None, dave_release=None, n_pyxtal_samples=1, **kwargs):
        super().__init__(**kwargs)

        # Import the necessary util function from the DAVE repository
        ensure_library_version("dave", DAVE_REPO_URL, dave_release)

        self.n_pyxtal_samples = n_pyxtal_samples

    @torch.no_grad()
    def __call__(self, states: TensorType["batch", "elements"]) -> TensorType["batch"]:
        """
        Forward pass of the proxy.

        Args:
            states (torch.Tensor): States to infer on. Shape:
                ``(batch, [n_elements + 1 + 6])``.

        Returns:
            torch.Tensor: Proxy energies. Shape: ``(batch,)``.
        """
        y = [self.evaluate_state(s) for s in states.numpy()]
        return tfloat(y, states.device, states.dtype)

    def evaluate_state(self, state) -> float:
        # Obtain composition, space group and lattice parameters from the state
        composition = state[:-7].astype("int32")
        space_group_idx = state[-7].astype("int32")
        a, b, c, alpha, beta, gamma = state[-6:]

        # Compute the list of elements and number of atoms per element
        elements = []
        atoms_per_element = []
        for atomic_number, nb_atoms in enumerate(composition):
            if nb_atoms > 0:
                elements.append(atomic_number)
                atoms_per_element.append(nb_atoms.item())

        # If something in the crystal is identified to be invalid, skip the process of
        # generating atom coordinated with PyXtal and return a default high energy for
        # invalid crystals
        if not is_valid_crystal(
            space_group_idx, elements, atoms_per_element, a, b, c, alpha, beta, gamma
        ):
            return self.ENERGY_INVALID_SAMPLE

        # Generate crystal samples with atom positions using PyXtal
        pyxtal_samples = self.generate_pyxtal_samples(
            space_group_idx, elements, atoms_per_element, a, b, c, alpha, beta, gamma
        )

        # Score the PyXtal samples using the proxy
        # TODO : If possible, process the PyXtal samples in batches
        if len(pyxtal_samples) > 0:
            pyxtal_sample_scores = []
            for sample in pyxtal_samples:
                # Convert the PyXtal crystal to the graph format expected by the model
                sample_graph = self.graph_from_pyxtal_crystal(sample)

                # Score the sample using the model
                sample_score = self.energy_from_graph(sample_graph)
                pyxtal_sample_scores.append(sample_score)

            global_sample_score = min(pyxtal_sample_scores)

        else:
            # PyXtal was unable to generate valid crystals given the state. Provide a
            # default bad score.
            global_sample_score = self.ENERGY_INVALID_SAMPLE

        return global_sample_score

    def graph_from_pyxtal_crystal(self, pyxtal_crystal):
        # Obtain util function from the DAVE repository
        from dave.utils.atoms_to_graph import AtomsToGraphs, pymatgen_structure_to_graph
        
        # Obtain util classes from OCD repository
        from ocdata.adsorbates import Adsorbate
        from ocdata.bulk_obj import Bulk
        #from ocdata.surfaces import Surface
        #from ocdata.combined import Combined
        
        # Obtain list of possible adsorbates
        pass                                                                                # TODO
        
        # Obtain a list of all symmetrically distinct surfaces of the bulk structure
        # created from the crystal
        crystal_bulk = pyxtal_crystal.to_pymatgen()                                                 # How to do this step?
        crystal_slabs = self.enumerate_bulk_surfaces(crystal_bulk)
        """
        crystal_slabs = []
        for miller_indices in get_symmetrically_distinct_miller_indices(
            crystal_bulk, MAX_MILLER
        ):
            slab_gen = SlabGenerator(
                initial_structure=bulk_struct,
                miller_index=millers,
                min_slab_size=7.0,
                min_vacuum_size=20.0,
                lll_reduce=False,
                center_slab=True,
                primitive=True,
                max_normal_search=1,
            )
            slabs = slab_gen.get_slabs(
                tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False
            )
            crystal_slabs.extend(slabs)
        """

        # Convert the PyXtal crystal to pymatgen and then to a graph format
        a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6.0,
            r_energy=False,
            r_forces=False,
            r_distances=True,
            r_edges=False,
        )
        graph = pymatgen_structure_to_graph(pyxtal_crystal.to_pymatgen(), a2g)
        return graph

    def energy_from_graph(self, graph):
        return 0  # TODO

    def generate_pyxtal_samples(
        self, space_group_idx, elements, atoms_per_element, a, b, c, alpha, beta, gamma
    ):
        # Instantiate space group
        space_group = Group(space_group_idx)

        # Create a lattice of the appropriate type
        lattice = Lattice.from_para(
            a, b, c, alpha, beta, gamma, ltype=space_group.lattice_type
        )

        pyxtal_samples = []
        for i in range(self.n_pyxtal_samples):
            try:
                crystal = pyxtal.pyxtal()
                crystal.from_random(
                    dim=3,
                    group=space_group,
                    species=elements,
                    numIons=atoms_per_element,
                    lattice=lattice,
                )
            except Comp_CompatibilityError:
                # PyXtal deems the composition incompatible with the space
                # group. PyXtal is sometimes wrong on this but, at any rate,
                # we can't general crystals using PyXtal for this sample. Skip.
                continue

            # Validate the produced crystal
            if crystal.valid:
                pyxtal_samples.append(crystal)

        return pyxtal_samples

    def enumerate_bulk_surfaces(self, bulk_structure):
        """
        This method wraps the method Bulk.enumerate_surfaces found at
        https://github.com/RolnickLab/ocp/blob/sample-adslab/ocdata/bulk_obj.py#L173C38-L173C38.
        This method implements the functionallity that we want but it is
        implemented in the context of the Bulk class which wasn't made to be used on a
        user-provided bulk structure.
        
        """
        # Obtain util classes from OCD repository
        from ocdata.adsorbates import Adsorbate
        from ocdata.bulk_obj import Bulk
        
        # Instantiate Bulk object.
        # The standard __init__ requires providing a database of bulk structures so
        # we override it to allow instantiation without having a database.
        Bulk.__init__ = lambda x: None
        bulk_instance = Bulk()
        bulk_instance.bulk_atoms = bulk_structure.to_ase_atoms()
        bulk_instance.mpid = None

        return bulk_instance.enumerate_surfaces()
