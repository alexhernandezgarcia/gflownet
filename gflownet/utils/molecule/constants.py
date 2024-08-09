from rdkit import Chem

# Edge and node feature names in a graph
step_feature_name = "step"
atomic_numbers_name = "atomic_numbers"
rotatable_edges_mask_name = "rotatable_edges"
rotation_affected_nodes_mask_name = "rotation_affected_nodes"
rotation_signs_name = "rotation_signs"

# Options for atoms featurization
AD_ATOM_TYPES = ("H", "C", "N", "O", "F", "S", "Cl")
ATOM_DEGREES = tuple(range(1, 7))
ATOM_HYBRIDIZATIONS = tuple(list(Chem.rdchem.HybridizationType.names.values()))
BOND_TYPES = tuple(
    [
        "FAKE",
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
)

# SMILES strings
AD_SMILES = "CC(C(=O)NC)NC(=O)C"
KETOROLAC_SMILES = "OC(=O)C1CCn2c1ccc2C(=O)c1ccccc1"
IBUPROFEN_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"


# Freely rotatable torsion angles
AD_FREE_TAS = ((0, 1, 2, 3), (0, 1, 6, 7))

# some selected SMILES strings used in the paper https://pubs.rsc.org/en/content/articlepdf/2024/DD/D4DD00023D
SELECTED_SMILES = [
    "O=C(c1ccccc1)c1ccc2c(c1)OCCOCCOCCOCCO2",
    "O=S(=O)(NN=C1CCCCCC1)c1ccc(Cl)cc1",
    "O=C(NC1CCCCC1)N1CCN(C2CCCCC2)CC1",
    "O=C(COc1ccc(Cl)cc1[N+](=O)[O-])N1CCCCCC1",
    "O=C(Nc1ccc(N2CCN(C(=O)c3ccccc3)CC2)cc1)c1cccs1",
    "O=[N+]([O-])/C(C(=C(Cl)Cl)N1CCN(Cc2ccccc2)CC1)=C1\\NCCN1Cc1ccc(Cl)nc1",
    "O=C(CSc1nnc(C2CC2)n1-c1ccccc1)Nc1ccc(N2CCOCC2)cc1",
    "O=C(Nc1ccccn1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)Nc1ccccn1",
    "O=C(NCCc1nnc2ccc(NCCCN3CCOCC3)nn12)c1ccccc1F",
    "O=C(CSc1cn(CCNC(=O)c2cccs2)c2ccccc12)NCc1ccccc1",
    "O=C(CCC(=O)N(CC(=O)NC1CCCCC1)Cc1cccs1)Nc1ccccn1",
    "S=C(c1ccc2c(c1)OCO2)N1CCOCC1",
    "O=Cc1cnc(N2CCN(c3ccccc3)CC2)s1",
    "O=[N+]([O-])c1ccc(NCc2ccc(Cl)cc2)nc1",
    "O=C(Nc1cc(C(F)(F)F)ccc1N1CCCCC1)c1ccncc1",
    "O=C(Nc1nnc(-c2ccccc2Cl)s1)C1CCN(S(=O)(=O)c2ccc(Cl)cc2)CC1",
    "O=C(CCNC(=O)c1ccccc1Cl)Nc1nc2ccccc2s1",
    "O=C(CCC(=O)Nc1ccccc1Cl)N/N=C/c1ccccc1",
    "O=C(COc1ccccc1)Nc1ccccc1C(=O)NCc1ccco1",
    "O=C(CNC(=O)c1cccs1)NCC(=O)OCc1ccc(Cl)cc1Cl",
    "O=C(NCc1ccccc1)c1onc(CSc2ccc(Cl)cc2)c1C(=O)NCC1CC1",
    "O=C(CN(C(=O)CCC(=O)Nc1ccccn1)c1ccc2c(c1)OCO2)NCc1ccco1",
    "O=[N+]([O-])c1ccc(N2CCNCC2)c(Cl)c1",
    "O=[N+]([O-])c1ccccc1S(=O)(=O)N1CCCCC1",
    "N#C/C(=C\\N1CCN(Cc2ccc3c(c2)OCO3)CC1)c1nc2ccccc2s1",
    "O=C(NNc1ccc([N+](=O)[O-])cc1)c1ccccc1Cl",
    "O=C(OCc1ccccc1Cl)c1ccccc1C(=O)c1ccccc1",
    "O=C(CN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1ccccc1)N1CCOCC1",
    "O=c1[nH]c2cc3c(cc2cc1CN(Cc1cccnc1)Cc1nnnn1Cc1ccco1)OCCO3",
    "O=C(CCNC(=O)C1CCN(S(=O)(=O)c2ccccc2)CC1)NC1CC1",
    "O=C(CN(c1ccc(F)cc1)S(=O)(=O)c1ccccc1)NCCSC1CCCCC1",
    "C=CCn1c(CSCc2ccccc2)nnc1SCC(=O)N1CCN(c2ccccc2)CC1",
    "C=COCCNC(=S)N1CCOCCOCCN(C(=S)NCCOC=C)CCOCC1",
    "O=S(=O)(c1cccc(Cl)c1Cl)N1CCCCC1",
    "O=C(Cn1ccccc1=O)c1cccs1",
    "FC(F)Sc1ccc(Nc2ncnc3nc[nH]c23)cc1",
    "O=c1[nH]c(SCCOc2ccccc2F)nc2ccccc12",
    "O=C(/C=C/c1ccccc1Cl)NCCN1CCOCC1",
    "N#CCCN1CCN(S(=O)(=O)c2ccc(S(=O)(=O)NC3CC3)cc2)CC1",
    "O=C(CSc1nnc(Cc2cccs2)n1-c1ccccc1)Nc1ccc2c(c1)OCCO2",
    "O=C(CNC(=O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1)OCC(=O)c1ccccc1",
    "O=C(CSc1nnc(CNC(=O)c2c(F)cccc2Cl)o1)NCc1ccc2c(c1)OCO2",
    "O=C(CNC(=S)N(Cc1ccc(F)cc1)C1CCCCC1)NCCN1CCOCC1",
    "C=CCn1c(CCNC(=O)c2cccs2)nnc1SCC(=O)Nc1cccc(F)c1",
    "Clc1cccc(CN2CCCCCC2)c1Cl",
    "O=C(Nc1cc(=O)c2ccccc2o1)N1CCCCC1",
    "O=S(=O)(c1ccccc1)c1nc(-c2ccco2)oc1N1CCOCC1",
    "O=C(CC1CCCCC1)NCc1ccco1",
    "O=C(c1cccc([N+](=O)[O-])c1)N1CCCN(C(=O)c2cccc([N+](=O)[O-])c2)CC1",
    "Fc1ccccc1OCCCCCN1CCCC1",
    "O=C(c1ccc(S(=O)(=O)NCc2ccco2)cc1)N1CCN(Cc2ccc3c(c2)OCO3)CC1",
    "N#Cc1ccc(NC(=O)COC(=O)CNC(=O)C2CCCCC2)cc1",
    "O=C(CCC(=O)OCC(=O)c1ccc(-c2ccccc2)cc1)Nc1cccc(Cl)c1",
    "O=C(COC(=O)CCC(=O)c1cccs1)Nc1ccc(S(=O)(=O)N2CCOCC2)cc1",
    "O=C(COC(=O)CCCNC(=O)c1ccc(Cl)cc1)NCc1ccccc1Cl",
    "C1CCC([NH2+]C2=NCCC2)CC1",
    "N#Cc1ccccc1S(=O)(=O)Nc1ccc2c(c1)OCCO2",
    "O=[N+]([O-])c1ccccc1S(=O)(=O)N1CCN(c2ccccc2)CC1",
    "O=S(=O)(NCc1ccccc1Cl)c1ccc(-n2cccn2)cc1",
    "O=C(CNS(=O)(=O)c1cccc2nsnc12)NC1CCCCC1",
    "O=C(c1cccc([N+](=O)[O-])c1)n1nc(-c2ccccc2)nc1NCc1ccccc1",
    "O=C(CN1CCN(c2ccccc2)CC1)NC(=O)NCc1ccco1",
    "O=C(CCCn1c(=O)c2ccccc2n(Cc2ccccc2)c1=O)NCc1ccco1",
    "O=C(COc1ccc(Cl)cc1)NCc1nnc(SCC(=O)N2CCCCCC2)o1",
    "O=C(NCc1ccccc1)c1onc(CSc2ccccn2)c1C(=O)NCc1ccccc1",
    "C=CCN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1cccc(C(=O)OCC(=O)Nc2ccccc2)c1",
    "O=S(=O)(N1CCCCCC1)N1CC[NH2+]CC1",
    "O=C1c2ccccc2C(=O)N1Cc1nn2c(-c3ccc(Cl)cc3)nnc2s1",
    "O=C(CN1C(=O)NC2(CCCC2)C1=O)Nc1ccc(F)c(F)c1F",
    "O=C(Cc1n[nH]c(=O)[nH]c1=O)N/N=C/c1ccccc1",
    "O=C(NCCSc1ccc(Cl)cc1)c1ccco1",
    "O=C(CN1CCN(Cc2ccccc2Cl)CC1)N/N=C/c1ccco1",
    "O=C(Nc1ccc(-c2csc(Nc3cccc(C(F)(F)F)c3)n2)cc1)c1cccc(C(F)(F)F)c1",
    "O=C(CCCCCN1C(=O)c2cccc3cccc(c23)C1=O)NCc1ccco1",
    "O=C(NCCCn1ccnc1)/C(=C\\c1cccs1)NC(=O)c1cccs1",
    "O=C(COC(=O)COc1ccccc1[N+](=O)[O-])Nc1ccc(S(=O)(=O)N2CCCCC2)cc1",
    "O=C(NCCCN1CCCC1=O)c1cc(NS(=O)(=O)c2ccc(F)cc2)cc(NS(=O)(=O)c2ccc(F)cc2)c1",
    "Clc1ccc(N2CCN(c3ncnc4c3oc3ccccc34)CC2)cc1",
    "O=C(Nc1c(Cl)ccc2nsnc12)c1cccnc1",
    "c1nc(COc2nsnc2N2CCOCC2)cs1",
    "O=C(C1CC1)N1CCN=C1SCc1ccccc1",
    "O=C(Cc1ccccc1)OCC[NH+]1CCOCC1",
    "O=C(CCSc1ccccc1)NCc1cccnc1",
    "O=C(CNC(=O)c1ccc(F)cc1)N/N=C/c1cn[nH]c1-c1ccccc1",
    "O=C1CCN(CCc2ccccc2)CCN1[C@H](CSc1ccccc1)Cc1ccccc1",
    "O=C(CCCCCn1c(=S)[nH]c2ccc(N3CCOCC3)cc2c1=O)NCc1ccc(Cl)cc1",
    "O=C(CCC(=O)OCCCC(F)(F)C(F)(F)F)NC1CCCCC1",
    "O=C(CN(Cc1ccco1)C(=O)CNS(=O)(=O)c1ccc(F)cc1)NCc1ccco1",
    "O=c1[nH]c(N2CCN(c3ccccc3)CC2)nc2c1CCC2",
    "O=C(Nc1ccc2c(c1)OCO2)c1cccs1",
    "O=C(Nc1cccc2ccccc12)N1CCN(c2ccccc2)CC1",
    "O=C(NC(=S)Nc1ccccn1)c1ccccc1",
    "O=C(Nc1ccc(Cl)cc1)c1cccc(S(=O)(=O)Nc2ccccn2)c1",
    "O=C(COc1cnc2ccccc2n1)NCCC1=CCCCC1",
    "O=C(NCCN1CCOCC1)c1ccc(/C=C2\\Sc3ccccc3N(Cc3ccc(F)cc3)C2=O)cc1",
    "O=C(NCCc1cccc(Cl)c1)c1ccc(OC2CCN(Cc3ccccn3)CC2)cc1",
    "C=CC[NH2+]CCOCCOc1ccccc1-c1ccccc1",
    "O=C(COC(=O)c1ccccc1NC(=O)c1ccco1)NCCC1=CCCCC1",
    "O=C(CNC(=S)N(Cc1ccccc1Cl)C1CCCC1)NCCCN1CCOCC1",
    "O=c1c2ccccc2nnn1Cc1ccccc1Cl",
    "S=C(Nc1ccccc1)N1CCCCCCC1",
    "O=C(Cn1ccc([N+](=O)[O-])n1)N1CCCc2ccccc21",
    "O=C(NS(=O)(=O)N1CCOCC1)C1=C(N2CCCC2)COC1=O",
    "O=C(CCCn1c(=O)[nH]c2ccsc2c1=O)NC1CCCCC1",
    "O=C(Cc1ccc(Cl)cc1)Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",
    "O=C1COc2ccc(C(=O)COC(=O)CCSc3ccccc3)cc2N1",
    "O=C(Nc1cc(F)cc(F)c1)c1ccc(NCCC[NH+]2CCCCCC2)c([N+](=O)[O-])c1",
    "O=C(CCCn1c(=O)[nH]c2cc(Cl)ccc2c1=O)NCCCN1CCN(c2ccc(F)cc2)CC1",
    "O=C(NCCN1CCN(C(=O)C(c2ccccc2)c2ccccc2)CC1)C(=O)Nc1ccccc1",
    "O=C(NCCCN1CCN(CCCNC(=O)c2ccc3c(c2)OCO3)CC1)c1ccc2c(c1)OCO2",
]
