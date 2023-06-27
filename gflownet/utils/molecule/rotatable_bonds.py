# Taken from https://pyxtal.readthedocs.io/en/latest/_modules/pyxtal/molecule.html.

from operator import itemgetter


def find_rotor_from_smile(smile):
    """
    Find the positions of rotatable bonds in the molecule.
    """

    def cleaner(list_to_clean, neighbors):
        """
        Remove duplicate torsion from a list of atom index tuples.
        """

        for_remove = []
        for x in reversed(range(len(list_to_clean))):
            ix0 = itemgetter(0)(list_to_clean[x])
            ix3 = itemgetter(3)(list_to_clean[x])
            # for i-j-k-l, we don't want i, l are the ending members
            # here C-C-S=O is not a good choice since O is only 1-coordinated
            if neighbors[ix0] > 1 and neighbors[ix3] > 1:
                for y in reversed(range(x)):
                    ix1 = itemgetter(1)(list_to_clean[x])
                    ix2 = itemgetter(2)(list_to_clean[x])
                    iy1 = itemgetter(1)(list_to_clean[y])
                    iy2 = itemgetter(2)(list_to_clean[y])
                    if [ix1, ix2] == [iy1, iy2] or [ix1, ix2] == [iy2, iy1]:
                        for_remove.append(y)
            else:
                for_remove.append(x)
        clean_list = []
        for i, v in enumerate(list_to_clean):
            if i not in set(for_remove):
                clean_list.append(v)
        return clean_list

    if smile in ["Cl-", "F-", "Br-", "I-", "Li+", "Na+"]:
        return []
    else:
        from rdkit import Chem

        smarts_torsion1 = "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]"
        smarts_torsion2 = "[*]~[^2]=[^2]~[*]"  # C=C bonds
        # smarts_torsion2="[*]~[^1]#[^1]~[*]" # C-C triples bonds, to be fixed

        mol = Chem.MolFromSmiles(smile)
        N_atom = mol.GetNumAtoms()
        neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        # make sure that the ending members will be counted
        neighbors[0] += 1
        neighbors[-1] += 1
        patn_tor1 = Chem.MolFromSmarts(smarts_torsion1)
        torsion1 = cleaner(list(mol.GetSubstructMatches(patn_tor1)), neighbors)
        patn_tor2 = Chem.MolFromSmarts(smarts_torsion2)
        torsion2 = cleaner(list(mol.GetSubstructMatches(patn_tor2)), neighbors)
        tmp = cleaner(torsion1 + torsion2, neighbors)
        torsions = []
        for t in tmp:
            (i, j, k, l) = t
            b = mol.GetBondBetweenAtoms(j, k)
            if not b.IsInRing():
                torsions.append(t)
        # if len(torsions) > 6: torsions[1] = (4, 7, 10, 15)
        return torsions
