from rdkit import Chem

# Edge and node feature names in DGL graph
atom_position_name = "pos"
atom_feature_name = "atom_features"
edge_feature_name = "edge_features"
step_feature_name = "step"
atomic_numbers_name = "atomic_numbers"

# Options for atoms featurization
ad_atom_types = ("H", "C", "N", "O")
atom_degrees = tuple(range(1, 7))
atom_hybridizations = tuple(list(Chem.rdchem.HybridizationType.names.values()))
bond_types = tuple([Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE])

# SMILES strings
ad_smiles = "CC(C(=O)NC)NC(=O)C"

# Freely rotatable torsion angles
ad_free_tas = ((0, 1, 2, 3), (0, 1, 6, 7))
