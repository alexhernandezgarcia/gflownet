import yaml
import pandas as pd
from typing import Iterable

# lattice offsets depending on the bravais centering
lattice_symbols = {
    "P": ["(0, 0, 0)"],
    "A": ["(0, 0, 0)", "(0, 1/2, 1/2)"],
    "B": ["(0, 0, 0)", "(1/2, 0, 1/2))"],
    "C": ["(0, 0, 0)", "(1/2, 1/2, 0)"],
    "I": ["(0, 0, 0)", "(1/2, 1/2, 1/2)"],
    "R": ["(0, 0, 0)"],
    "H": ["(0, 0, 0)", "(2/3, 1/3, 1/3)", "(1/3, 2/3, 2/3)"],
    "F": ["(0, 0, 0)", "(0, 1/2, 1/2)", "(1/2, 0, 1/2)", "(1/2, 1/2, 0)"],
    "p": ["(0, 0, 0)"],
    "c": ["(0, 0, 0)", "(1/2, 1/2, 0)"],
}


def parse_wyckoff_csv():
    """Originally implemented in the spglib package, see https://github.com/spglib/spglib
        Adapted following our own needs.
    Uses:    Wyckoff.csv: file taken from above repo.
             spg.csv: file mapping hall numbers (column index 0) to spacegroup numbers (column index 4).

    There are 530 data sets. For one example:

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
    with open("gflownet/envs/crystals/wyckoff.csv", "r") as fp:
        wyckoff_file = parse_wyckoff_csv(fp)
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
            n_pos *= len(lattice_symbols[wyckoff[i]["symbol"][0]])
            assert n_pos == w["multiplicity"]

        # only keeping the standard settings
        df_hall = pd.read_csv("gflownet/envs/crystals/spg.csv", header=None)
        hall_numbers = []
        for sg_number in range(1, 231):
            hall_numbers.append(df_hall[(df_hall[4] == sg_number)].index[0])
        spacegroups = [spacegroups[sg_number] for sg_number in hall_numbers]

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
        """get offset positions encoded as: ["(0,0,0)","(1/2,1/2,1/2)"]

        Returns
        -------
        Iterable
            offset positions
        """
        return self.offset

    def get_algebraic(self) -> Iterable:
        """get algebraic positions encoded as: ["(x,y,z)","(-x,-y,-z)"]

        Returns
        -------
        Iterable
            algebraic positions
        """
        return self.algebraic

    def get_all_positions(self) -> Iterable:
        """get all Wyckoff positions, i.e. outer product over offsets and algebraic contributions. The number of positions equals the multiplicity.

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
        return hash((self.offset, self.algebraic))

    def __eq__(self, __value: object) -> bool:
        return self.__hash__() == __value.__hash__()

    def __str__(self) -> str:
        wyckoff_str = "Multiplicity: {}\n".format(
            len(self.offset) * len(self.algebraic)
        )
        wyckoff_str += "Offset: "
        wyckoff_str += str(list(self.offset)) + "\n"
        wyckoff_str += "Algebraic: "
        wyckoff_str += str(list(self.algebraic))
        return wyckoff_str


spacegroups = parse_wyckoff_csv()

for i, w in enumerate(spacegroups):
    for p in w["wyckoff"]:
        pure_trans = lattice_symbols[w["symbol"][0]]
        p["positions"] = Wyckoff(pure_trans, p["positions"])

wyckoff_positions = []
for sg in spacegroups:
    for w in sg["wyckoff"]:
        wyckoff_positions.append(w["positions"])

wyckoff_positions = list(set(wyckoff_positions))
wyckoff_mapping = {wyckoff_positions[i]: i + 1 for i in range(len(wyckoff_positions))}
wyckoff_dict = {
    i + 1: wyckoff_positions[i].as_dict() for i in range(len(wyckoff_positions))
}

with open("gflownet/envs/crystals/wyckoff.yaml", "w") as fp:
    yaml.dump(wyckoff_dict, fp)

with open("gflownet/envs/crystals/space_groups.yaml") as fp:
    spacegroup_dict = yaml.safe_load(fp)

for i in range(0, 230):
    spacegroup_dict[i + 1]["wyckoff_positions"] = [
        wyckoff_mapping[w["positions"]] for w in spacegroups[i]["wyckoff"]
    ]

with open("gflownet/envs/crystals/space_groups_wyckoff.yaml", "w") as fp:
    yaml.dump(spacegroup_dict, fp)
