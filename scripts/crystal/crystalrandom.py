from typing import List

import numpy as np


def generate_random_crystals(
    n_samples: int,
    elements: list[int],
    min_elements: int,
    max_elements: int,
    max_atoms: int,
    max_atom_i: int,
    space_groups: list[int],
    min_length: float,
    max_length: float,
    min_angle: float,
    max_angle: float,
):
    samples = []
    for _ in range(n_samples):
        # Elements of composition
        composition = [0] * len(elements)
        n_elements = np.random.randint(low=min_elements, high=max_elements + 1)
        elements_indices = np.random.randint(low=0, high=len(elements), size=n_elements)
        elements_selected = [elements[int(idx)] for idx in elements_indices]
        # Atoms per element
        for idx, el in enumerate(elements_selected):
            n_elements_missing = n_elements - idx
            n_atoms_max = min(
                max_atom_i, max_atoms - sum(composition) - n_elements_missing
            )
            n_atoms_el = np.random.randint(low=1, high=n_atoms_max + 1)
            composition[elements.index(el)] = n_atoms_el

        # Space group
        space_group = [0, 0, np.random.permutation(space_groups)[0]]
        # Lattice parameters
        lengths = list(np.random.uniform(low=min_length, high=max_length, size=3))
        angles = list(np.random.uniform(low=min_angle, high=max_angle, size=3))
        # State
        state = [2] + composition + space_group + lengths + angles
        samples.append(state)
    return samples


def generate_random_crystals_uniform(
    n_samples: int,
    elements: list[int],
    min_elements: int,
    max_elements: int,
    max_atoms: int,
    max_atom_i: int,
    space_groups: list[int],
    min_length: float,
    max_length: float,
    min_angle: float,
    max_angle: float,
):
    samples = []
    for _ in range(n_samples):
        # Elements of composition

        n_elements = np.random.randint(low=min_elements, high=max_elements + 1)
        elements_indices = np.random.randint(low=0, high=len(elements), size=n_elements)
        elements_selected = [elements[int(idx)] for idx in elements_indices]
        # Atoms per element
        done = False
        while not done:
            composition = [0] * len(elements)
            for el in elements_selected:
                n_atoms_el = np.random.randint(low=1, high=max_atom_i + 1)
                composition[elements.index(el)] = n_atoms_el
            if sum(composition) <= max_atoms:
                done = True

        # Space group
        space_group = [0, 0, np.random.permutation(space_groups)[0]]
        # Lattice parameters
        lengths = list(np.random.uniform(low=min_length, high=max_length, size=3))
        angles = list(np.random.uniform(low=min_angle, high=max_angle, size=3))
        # State
        state = [2] + composition + space_group + lengths + angles
        samples.append(state)
    return samples


# samples = generate_random_crystals(
#     n_samples=10,
#     elements=[1, 3, 6, 7, 8, 9, 12, 14, 15, 16, 17, 26],
#     min_elements=2,
#     max_elements=5,
#     max_atoms=16,
#     max_atom_i=50,
#     space_groups=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     min_length=0.9,
#     max_length=100.0,
#     min_angle=50.0,
#     max_angle=150.0,
# )
#
# for sample in samples:
#     print(sample)
