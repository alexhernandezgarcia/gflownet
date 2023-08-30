CRYSTAL_SYSTEMS = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "trigonal",
    6: "hexagonal",
    7: "cubic",
}
LATTICE_SYSTEMS = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "rhombohedral",
    6: "hexagonal",
    7: "cubic",
}
CRYSTAL_LATTICE_SYSTEMS = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "trigonal-rhombohedral",
    6: "trigonal-hexagonal",
    7: "hexagonal",
    8: "cubic",
}
RHOMBOHEDRAL_SPACE_GROUPS_WIKIPEDIA = ["R3", "R-3", "R32", "R3m", "R3c", "R-3m", "R-3c"]
POINT_SYMMETRIES = {
    1: "enantiomorphic-polar",
    2: "centrosymmetric",
    3: "polar",
    4: "enantiomorphic",
    5: "non-centrosymmetric",
}
# Dictionary summarising Wikipedia's crystal classes table
# See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_classes
# See: http://pd.chem.ucl.ac.uk/pdnn/symm2/group32.htm
# Each item in the dictionary contains a list of:
# - crystal class name
# - crystal system
# - list of point groups in Hermann-Mauguin format as in pymatgen
# - point symmetry
CRYSTAL_CLASSES_WIKIPEDIA = {
    1: ["pedial", "triclinic", ["1"], "enantiomorphic-polar"],
    2: ["pinacoidal", "triclinic", ["-1"], "centrosymmetric"],
    3: ["sphenoidal", "monoclinic", ["2"], "enantiomorphic-polar"],
    4: ["domatic", "monoclinic", ["m"], "polar"],
    5: ["prismatic", "monoclinic", ["2/m"], "centrosymmetric"],
    6: ["rhombic-disphenoidal", "orthorhombic", ["222"], "enantiomorphic"],
    7: ["rhombic-pyramidal", "orthorhombic", ["mm2"], "polar"],
    8: ["rhombic-dipyramidal", "orthorhombic", ["mmm"], "centrosymmetric"],
    9: ["tetragonal-pyramidal", "tetragonal", ["4"], "enantiomorphic-polar"],
    10: ["tetragonal-disphenoidal", "tetragonal", ["-4"], "non-centrosymmetric"],
    11: ["tetragonal-dipyramidal", "tetragonal", ["4/m"], "centrosymmetric"],
    12: ["tetragonal-trapezohedral", "tetragonal", ["422"], "enantiomorphic"],
    13: ["ditetragonal-pyramidal", "tetragonal", ["4mm"], "polar"],
    14: [
        "tetragonal-scalenohedral",
        "tetragonal",
        ["-42m", "-4m2"],
        "non-centrosymmetric",
    ],
    15: ["ditetragonal-dipyramidal", "tetragonal", ["4/mmm"], "centrosymmetric"],
    16: ["trigonal-pyramidal", "trigonal", ["3"], "enantiomorphic-polar"],
    17: ["rhombohedral", "trigonal", ["-3"], "centrosymmetric"],
    18: [
        "trigonal-trapezohedral",
        "trigonal",
        ["32", "321", "312"],
        "enantiomorphic",
    ],
    19: ["ditrigonal-pyramidal", "trigonal", ["3m", "3m1", "31m"], "polar"],
    20: [
        "ditrigonal-scalenohedral",
        "trigonal",
        ["-3m", "-3m1", "-31m"],
        "centrosymmetric",
    ],
    21: ["hexagonal-pyramidal", "hexagonal", ["6"], "enantiomorphic-polar"],
    22: ["trigonal-dipyramidal", "hexagonal", ["-6"], "non-centrosymmetric"],
    23: ["hexagonal-dipyramidal", "hexagonal", ["6/m"], "centrosymmetric"],
    24: ["hexagonal-trapezohedral", "hexagonal", ["622"], "enantiomorphic"],
    25: ["dihexagonal-pyramidal", "hexagonal", ["6mm"], "polar"],
    26: [
        "ditrigonal-dipyramidal",
        "hexagonal",
        ["-6m2", "-62m"],
        "non-centrosymmetric",
    ],
    27: ["dihexagonal-dipyramidal", "hexagonal", ["6/mmm"], "centrosymmetric"],
    28: ["tetartoidal", "cubic", ["23"], "enantiomorphic"],
    29: ["diploidal", "cubic", ["m-3"], "centrosymmetric"],
    30: ["gyroidal", "cubic", ["432"], "enantiomorphic"],
    31: ["hextetrahedral", "cubic", ["-43m"], "non-centrosymmetric"],
    32: ["hexoctahedral", "cubic", ["m-3m"], "centrosymmetric"],
}
