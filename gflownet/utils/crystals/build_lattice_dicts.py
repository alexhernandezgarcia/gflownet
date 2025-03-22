"""
Reads a dictionary with crystal classes and their point groups, and a dictionary with
the point group of each space group, and outputs a dictionary containing both the point
groups and space groups of each crystal class.
"""

import sys
from argparse import ArgumentParser

import numpy as np
import yaml
from lattice_constants import (
    CRYSTAL_CLASSES_WIKIPEDIA,
    CRYSTAL_LATTICE_SYSTEMS,
    CRYSTAL_SYSTEMS,
    POINT_SYMMETRIES,
    RHOMBOHEDRAL_SPACE_GROUPS_WIKIPEDIA,
)
from pymatgen.symmetry.groups import (
    PointGroup,
    SpaceGroup,
    SymmetryGroup,
    sg_symbol_from_int_number,
)

N_SPACE_GROUPS = 230


def parsed_args():
    """
    Parse and returns command-line args

    Returns
    -------
    argparse.Namespace
        the parsed arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--output_sg_yaml",
        default=None,
        type=str,
        help="Output space groups YAML dict",
    )
    parser.add_argument(
        "--output_cls_yaml",
        default=None,
        type=str,
        help="Output crystal-lattice system YAML dict",
    )
    parser.add_argument(
        "--output_ps_yaml",
        default=None,
        type=str,
        help="Output point symmetry YAML dict",
    )
    parser.add_argument(
        "--print",
        default=False,
        action="store_true",
        help="If True, print progress",
    )
    return parser.parse_args()


def _crystal_lattice_system_index(crystal_lattice_system):
    for idx, crystal_lattice_system_iter in CRYSTAL_LATTICE_SYSTEMS.items():
        if crystal_lattice_system == crystal_lattice_system_iter:
            return idx
    raise ValueError(
        f"Crystal-lattice system {crystal_lattice_system} not found in crystal-lattice systems dictionary"
    )


def _point_symmetry_index(point_symmetry):
    for idx, point_symmetry_iter in POINT_SYMMETRIES.items():
        if point_symmetry == point_symmetry_iter:
            return idx
    raise ValueError(
        f"Point symmetry {point_symmetry} not found in point symmetries dictionary"
    )


def _point_group_to_crystal_class_and_point_symmetry(point_group):
    for crystal_class_idx, pg_list in CRYSTAL_CLASSES_WIKIPEDIA.items():
        if point_group in pg_list[2]:
            crystal_class = pg_list[0]
            point_symmetry = pg_list[3]
            point_symmetry_idx = _point_symmetry_index(point_symmetry)
            return (
                crystal_class_idx,
                crystal_class,
                point_symmetry_idx,
                point_symmetry,
            )
    raise ValueError(f"Point group {point_group} not found in point group dictionary")


def _crystal_system_and_space_group_to_lattice_system(crystal_system, sg_symbol):
    if crystal_system == "trigonal":
        if sg_symbol in RHOMBOHEDRAL_SPACE_GROUPS_WIKIPEDIA:
            return "rhombohedral"
        else:
            return "hexagonal"
    else:
        return crystal_system


def _crystal_system_and_space_groups_to_lattice_system(
    crystal_system, sg_symbol, sg_symbol_rhombohedral
):
    if crystal_system == "trigonal":
        if sg_symbol_rhombohedral != sg_symbol:
            return "rhombohedral"
        else:
            return "hexagonal"
    else:
        return crystal_system


if __name__ == "__main__":
    # Arguments
    args = parsed_args()

    # Make space group dictionary
    space_groups_dict = {}
    for idx in range(1, N_SPACE_GROUPS + 1):
        sg_symbol = sg_symbol_from_int_number(idx, hexagonal=True)
        sg_symbol_rhombohedral = sg_symbol_from_int_number(idx, hexagonal=False)
        sg = SpaceGroup(sg_symbol)
        sg_hmsymbol = sg.symbol
        point_group = sg.point_group
        crystal_system = sg.crystal_system
        (
            crystal_class_idx,
            crystal_class,
            point_symmetry_idx,
            point_symmetry,
        ) = _point_group_to_crystal_class_and_point_symmetry(point_group)
        lattice_system = _crystal_system_and_space_group_to_lattice_system(
            crystal_system, sg_symbol_rhombohedral
        )
        lattice_system_pymatgen = _crystal_system_and_space_groups_to_lattice_system(
            crystal_system, sg_symbol, sg_symbol_rhombohedral
        )
        assert lattice_system == lattice_system_pymatgen
        if crystal_system != lattice_system:
            crystal_lattice_system = f"{crystal_system}-{lattice_system}"
        else:
            crystal_lattice_system = crystal_system
        crystal_lattice_system_idx = _crystal_lattice_system_index(
            crystal_lattice_system
        )
        # Update dictionary
        space_groups_dict.update(
            {
                idx: {
                    "full_symbol": sg_hmsymbol,
                    "symbol": sg_symbol_rhombohedral,
                    "crystal_system": crystal_system,
                    "lattice_system": lattice_system,
                    "crystal_lattice_system_idx": crystal_lattice_system_idx,
                    "point_symmetry_idx": point_symmetry_idx,
                    "point_symmetry": point_symmetry,
                    "point_group": point_group,
                    "crystal_class": crystal_class,
                }
            }
        )
        # Print
        if args.print:
            print(f"International number: {idx}")
            for k, v in space_groups_dict[idx].items():
                print(f"{k}: {v}")
            print("---")
    # Save YAML
    if args.output_sg_yaml:
        with open(args.output_sg_yaml, "w") as f:
            yaml.dump(space_groups_dict, f)

    # Make crystal-lattice system dictionary
    crystal_lattice_system_dict = {}
    for idx, crystal_lattice_system in CRYSTAL_LATTICE_SYSTEMS.items():
        crystal_system = crystal_lattice_system.split("-")[0]
        lattice_system = crystal_lattice_system.split("-")[-1]
        space_groups = []
        point_symmetries = []
        for sg, sg_dict in space_groups_dict.items():
            if (
                sg_dict["crystal_system"] == crystal_system
                and sg_dict["lattice_system"] == lattice_system
            ):
                space_groups.append(sg)
                point_symmetry_index = _point_symmetry_index(sg_dict["point_symmetry"])
                if point_symmetry_index not in point_symmetries:
                    point_symmetries.append(point_symmetry_index)
        # Update dictionary
        crystal_lattice_system_dict.update(
            {
                idx: {
                    "crystal_system": crystal_system,
                    "lattice_system": lattice_system,
                    "point_symmetries": point_symmetries,
                    "space_groups": space_groups,
                }
            }
        )
        # Print
        if args.print:
            print(f"Crystal-lattice system index: {idx}")
            for k, v in crystal_lattice_system_dict[idx].items():
                print(f"{k}: {v}")
            print("---")
    # Save YAML
    if args.output_cls_yaml:
        with open(args.output_cls_yaml, "w") as f:
            yaml.dump(crystal_lattice_system_dict, f)

    # Make point symmetry dictionary
    point_symmetry_dict = {}
    for idx, point_symmetry in POINT_SYMMETRIES.items():
        space_groups = []
        crystal_lattice_systems = []
        for sg, sg_dict in space_groups_dict.items():
            if sg_dict["point_symmetry"] == point_symmetry:
                space_groups.append(sg)
                crystal_system = sg_dict["crystal_system"]
                lattice_system = sg_dict["lattice_system"]
                if crystal_system != lattice_system:
                    crystal_lattice_system = f"{crystal_system}-{lattice_system}"
                else:
                    crystal_lattice_system = crystal_system
                crystal_lattice_system_idx = _crystal_lattice_system_index(
                    crystal_lattice_system
                )
                if crystal_lattice_system_idx not in crystal_lattice_systems:
                    crystal_lattice_systems.append(crystal_lattice_system_idx)
        # Update dictionary
        point_symmetry_dict.update(
            {
                idx: {
                    "point_symmetry": point_symmetry,
                    "crystal_lattice_systems": crystal_lattice_systems,
                    "space_groups": space_groups,
                }
            }
        )
        # Print
        if args.print:
            print(f"Point symmetry index: {idx}")
            for k, v in point_symmetry_dict[idx].items():
                print(f"{k}: {v}")
            print("---")
    # Save YAML
    if args.output_cls_yaml:
        with open(args.output_ps_yaml, "w") as f:
            yaml.dump(point_symmetry_dict, f)

    # Exit
    sys.exit()
