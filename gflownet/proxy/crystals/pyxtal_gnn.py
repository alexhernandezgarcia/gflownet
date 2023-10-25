import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.msg import Comp_CompatibilityError
from pyxtal.symmetry import Group


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
    if a + b + c >= 360:
        return False
    elif a + b - c <= 0:
        return False
    elif a + c - b <= 0:
        return False
    elif b + c + a <= 0:
        return False

    return True


class PyxtalGNN(Proxy):
    """
    Proxy using a GNN model to evaluate samples.

    This proxy assumes that the samples do not contain atom coordinates and
    therefore uses PyXtal to sample atom coordinates before feeding the samples
    to the GNN model.
    """

    ENERGY_INVALID_SAMPLE = 10

    def __init__(self, n_pyxtal_samples=1, **kwargs):
        super().__init__(**kwargs)
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
                sample_graph = self.graph_from_pyxtal(sample)

                # Score the sample using the model
                sample_score = self.energy_from_graph(sample_graph)
                pyxtal_sample_scores.append(sample_score)

            global_sample_score = min(pyxtal_sample_scores)

        else:
            # PyXtal was unable to generate valid crystals given the state. Provide a
            # default bad score.
            global_sample_score = self.ENERGY_INVALID_SAMPLE

        return global_sample_score

    def graph_from_pyxtal(self, pyxtal_crystal):
        return None #TODO

    def energy_from_graph(self, graph):
        return 0 #TODO

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
                    lattice=lattice
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


    def pyxtal2proxy(self, pyxtal_crystal):
        # TODO
        return None

