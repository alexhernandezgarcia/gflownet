"""
A simple script to determine which space group symbols are different in pyxtal and
pymatgen.
"""
from argparse import ArgumentParser

from pymatgen.symmetry.groups import (
    PointGroup,
    SpaceGroup,
    SymmetryGroup,
    sg_symbol_from_int_number,
)
from pyxtal.symmetry import Group

N_SYMMETRY_GROUPS = 230

for idx in range(1, N_SYMMETRY_GROUPS + 1):
    sg_int = sg_symbol_from_int_number(idx)
    sg_pymatgen = SpaceGroup(sg_int)
    sg_pyxtal = Group(idx)
    if sg_pymatgen.symbol != sg_pyxtal.symbol or sg_pyxtal.number != idx:
        print(f"idx: {idx}")
        print(f"sg_int: {sg_int}")
        print(f"sg_pyxtal_number: {sg_pyxtal.number}")
        print(f"pymatgen symbol: {sg_pymatgen.symbol}")
        print(f"pyxtal symbol: {sg_pyxtal.symbol}")
        print("---")
