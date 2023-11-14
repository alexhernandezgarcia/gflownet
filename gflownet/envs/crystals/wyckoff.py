import yaml
import pandas as pd
from typing import Iterable

# lattice offsets depending on the Bravais centering
# most are trivial to understand (see Bravais lattices)
# H centering is a bit tricky: it corresponds to
# the orthorombic lattice in hexagonal axes setting.
lattice_offset = {
    "P": ["(0, 0, 0)"],  # primitive
    "A": ["(0, 0, 0)", "(0, 1/2, 1/2)"],
    "B": ["(0, 0, 0)", "(1/2, 0, 1/2))"],
    "C": ["(0, 0, 0)", "(1/2, 1/2, 0)"],
    "I": ["(0, 0, 0)", "(1/2, 1/2, 1/2)"],
    "R": ["(0, 0, 0)"],
    "H": [
        "(0, 0, 0)",
        "(2/3, 1/3, 1/3)",
        "(1/3, 2/3, 2/3)",
    ],  # rhombohedric in hexagonal axes
    "F": [
        "(0, 0, 0)",
        "(0, 1/2, 1/2)",
        "(1/2, 0, 1/2)",
        "(1/2, 1/2, 0)",
    ],  # Face centered
    "p": ["(0, 0, 0)"],
    "c": ["(0, 0, 0)", "(1/2, 1/2, 0)"],
}


def parse_wyckoff_csv():
    """Originally implemented in the spglib package, see https://github.com/spglib/spglib
        Adapted following our own needs.
    Uses:    Wyckoff.csv: file taken from above repo.
             spg.csv: file mapping hall numbers (column index 0) to spacegroup numbers (column index 4).

    There are 530 entries. For one example:

    9:C 1 2 1:::::::
    ::4:c:1:(x,y,z):(-x,y,-z)::
    ::2:b:2:(0,y,1/2):::
    ::2:a:2:(0,y,0):::

    Any number in first column corresponds to the hall number.
    The 230 space groups can be represented in cartesian coordinates following 530 conventional settings,
    depending on the origin and orientation of the axes. We decide to take the one convention per spacegroup,
    following the same procedure as the Bilbao Crystallographic server:

        "These are specific settings of space groups that coincide with the conventional space-group
        descriptions found in Volume A of International Tables for Crystallography (referred to as ITA).
        For space groups with more than one description in ITA, the following settings are chosen as
        standard: unique axis b setting, cell choice 1 for monoclinic groups, hexagonal axes setting
        for rhombohedral groups, and origin choice 2 (origin in -1) for the centrosymmetric groups listed
        with respect to two origins in ITA."

    For more info see https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-def-ita_settings?what=wp
    and the International Tables.

    The next lines in the csv files specify the individual positions in form 4:c:1:(x,y,z):(-x,y,-z)::,
    corresponding to the multiplicity, Wyckoff letter, site symmetry and algebraic terms. If more than 4 terms are present,
    they are written on multiple lines.
    """
    wyckoff_file = open("gflownet/envs/crystals/wyckoff.csv", "r")
    rowdata = []
    points = []
    hP_nums = [433, 436, 444, 450, 452, 458, 460]
    for i, line in enumerate(wyckoff_file):
        if line.strip() == "end of data":
            break
        rowdata.append(line.strip().split(":"))

        # 2:P -1  ::::::: <-- store line number if first element is number
        if rowdata[-1][0].isdigit():
            points.append(i)
    points.append(i)

    wyckoff = []
    for i in range(len(points) - 1):  # 0 to 529
        symbol = rowdata[points[i]][1]  # e.g. "C 1 2 1"
        if i + 1 in hP_nums:
            symbol = symbol.replace("R", "H", 1)
        wyckoff.append({"symbol": symbol.strip()})

    # When the number of positions is larger than 4,
    # the positions are written in the next line.
    # So those positions are connected.
    for i in range(len(points) - 1):
        count = 0
        wyckoff[i]["wyckoff"] = []
        for j in range(points[i] + 1, points[i + 1]):
            # Hook if the third element is a number (multiplicity), e.g.,
            #
            # 232:P 2/b 2/m 2/b:::::::  <- ignored
            # ::8:r:1:(x,y,z):(-x,y,-z):(x,-y+1/2,-z):(-x,-y+1/2,z)
            # :::::(-x,-y,-z):(x,-y,z):(-x,y+1/2,z):(x,y+1/2,-z)  <- ignored
            # ::4:q:..m:(x,0,z):(-x,0,-z):(x,1/2,-z):(-x,1/2,z)
            # ::4:p:..2:(0,y,1/2):(0,-y+1/2,1/2):(0,-y,1/2):(0,y+1/2,1/2)
            # ::4:o:..2:(1/2,y,0):(1/2,-y+1/2,0):(1/2,-y,0):(1/2,y+1/2,0)
            # ...
            if rowdata[j][2].isdigit():
                pos = []
                w = {
                    "letter": rowdata[j][3].strip(),
                    "multiplicity": int(rowdata[j][2]),
                    "site_symmetry": rowdata[j][4].strip(),
                    "positions": pos,
                }
                wyckoff[i]["wyckoff"].append(w)

                for k in range(4):
                    if rowdata[j][k + 5]:  # check if '(x,y,z)' or ''
                        count += 1
                        pos.append(rowdata[j][k + 5])
            else:
                for k in range(4):
                    if rowdata[j][k + 5]:
                        count += 1
                        pos.append(rowdata[j][k + 5])

        # assertion
        for w in wyckoff[i]["wyckoff"]:
            n_pos = len(w["positions"])
            n_pos *= len(lattice_offset[wyckoff[i]["symbol"][0]])
            assert n_pos == w["multiplicity"]

    # only keeping the standard settings
    df_hall = pd.read_csv("gflownet/envs/crystals/spg.csv", header=None)
    hall_numbers = []
    for sg_number in range(1, 231):
        hall_numbers.append(df_hall[(df_hall[4] == sg_number)].index[0])

    wyckoff = [wyckoff[sg_number] for sg_number in hall_numbers]

    return wyckoff


class Wyckoff:
    """Simple class representing a Wyckoff position. Only defined by its offset and algebraic terms."""

    def __init__(self, offset: Iterable, algebraic: Iterable) -> None:
        """
        Parameters
        ----------
        offset : Iterable
            offset positions encoded as follows, e.g. ["(0,0,0)","(1/2,1/2,1/2)"]
        algebraic : Iterable
            algebraic positions encoded as follows, e.g. ["(x,y,z)","(-x,-y,-z)"]
        """
        self.offset = frozenset(offset)
        self.algebraic = frozenset(algebraic)

    def get_multiplicity(self) -> int:
        """Get Wyckoff multiplicity: number of sites

        Returns
        -------
        int
            multiplicity
        """
        return len(self.offset) * len(self.algebraic)

    def get_offset(self) -> Iterable:
        """Get offset positions encoded as: ["(0,0,0)","(1/2,1/2,1/2)"]

        Returns
        -------
        Iterable
            offset positions
        """
        return self.offset

    def get_algebraic(self) -> Iterable:
        """Get algebraic positions encoded as: ["(x,y,z)","(-x,-y,-z)"]

        Returns
        -------
        Iterable
            algebraic positions
        """
        return self.algebraic

    def get_all_positions(self) -> Iterable:
        """Get all Wyckoff positions, i.e. outer product over offsets and algebraic contributions. The number of positions equals the multiplicity.

        Returns
        -------
        Iterable
            All Wyckoff positions.
        """

        w_pos = []
        for ofs in self.offset:
            ofs = ofs.strip("()").split(",")
            for a in self.algebraic:
                a = a.strip("()").split(",")
                w_pos.append(tuple([x + y for x, y in zip(ofs, a)]))
        return w_pos

    def as_dict(self) -> dict:
        """Dictionary representation of the wyckoff position

        Returns
        -------
        dict
            keys: multiplicity, offset, algebraic and positions.
        """
        return {
            "multiplicity": self.get_multiplicity(),
            "offset": list(self.get_offset()),
            "algebraic": list(self.get_algebraic()),
            "positions": [list(p) for p in self.get_all_positions()],
        }

    def __hash__(self) -> int:
        """Hash function to compare equivalent Wyckoff positions"""
        return hash((self.offset, self.algebraic))

    def __eq__(self, __value: object) -> bool:
        """Two Wyckoff positions are same if and only if they have the same set of
        offsets and algebraic terms, order invariant."""
        return self.__hash__() == __value.__hash__()

    def __str__(self) -> str:
        """Simple string representation of a Wyckoff position."""
        wyckoff_str = "Multiplicity: {}\n".format(
            len(self.offset) * len(self.algebraic)
        )
        wyckoff_str += "Offset: "
        wyckoff_str += str(list(self.offset)) + "\n"
        wyckoff_str += "Algebraic: "
        wyckoff_str += str(list(self.algebraic))
        return wyckoff_str


# get all spacegroups in stadard settings
spacegroups = parse_wyckoff_csv()


# convert to Wyckoff objects
wyckoff_positions = []
for spacegroup in spacegroups:
    for wyckoff_data in spacegroup["wyckoff"]:
        pure_trans = lattice_offset[spacegroup["symbol"][0]]  # get space group letter
        wyckoff_data["positions"] = Wyckoff(pure_trans, wyckoff_data["positions"])
        wyckoff_positions.append(wyckoff_data["positions"])


# create a mapping between unique idx and unique Wyckoff position
wyckoff_positions = list(set(wyckoff_positions))
wyckoff2idx = {
    wyckoff_position: idx + 1 for idx, wyckoff_position in enumerate(wyckoff_positions)
}
idx2wyckoff = {
    idx + 1: wyckoff_position.as_dict()
    for idx, wyckoff_position in enumerate(wyckoff_positions)
}


# create yaml file for unique Wyckoff indices
with open("gflownet/envs/crystals/wyckoff.yaml", "w") as fp:
    yaml.dump(idx2wyckoff, fp)


# Update yaml file defining the spacegroups
with open("gflownet/envs/crystals/space_groups.yaml") as fp:
    spacegroup_yaml_dict = yaml.safe_load(fp)

for sg_number in range(1, 231):
    spacegroup_yaml_dict[sg_number]["wyckoff_positions"] = [
        wyckoff2idx[wyckoff_data["positions"]]
        for wyckoff_data in spacegroups[sg_number - 1]["wyckoff"]
    ]

with open("gflownet/envs/crystals/space_groups_wyckoff.yaml", "w") as fp:
    yaml.dump(spacegroup_yaml_dict, fp)
