from pyxtal.symmetry import Group

_pyxtal_space_group_cache = {}
_pyxtal_check_compatible_cache = {}


def get_space_group(group_index):
    """
    Returns a PyxTal Group object representing a crystal space group

    This methods includes a lazy caching mechanism since the instantiation of
    a pyxtal.symmetry.Group object is expensive.

    Args
    ----
    group_index : int
        Index (starting at 1) of the space group

    Returns
    -------
    pyxtal.symmetry.Group
        Requested space group
    """
    if group_index not in _pyxtal_space_group_cache:
        _pyxtal_space_group_cache[group_index] = Group(group_index)
    return _pyxtal_space_group_cache[group_index]


def space_group_check_compatible(group_index, composition):
    """
    Determines if a given atom composition is compatible with a space group

    This methods internally relies on pyxtal.symmetry.Group.check_compatible()
    to determine if a composition and a space group are compatible but this
    method includes a caching mechanism since the call to check_compatible()
    is expensive.

    Args
    ----
    group_index : int
        Index (starting at 1) of the space group

    composition : list of ints
        Atom composition. Each element in the list corresponds to the number
        of atoms of a distinct element in the crystal conventional cell

    Returns
    -------
    is_compatible : bool
        True if the composition and space group are compatible. False
        otherwise.
    """
    # Get a tuple version of composition to ensure immutability and,
    # therefore, allow dictionary indexing by composition. Ensure the elements
    # in the tuple are sorted to improve cache hit rate since it doesn't
    # affect the validity of a composition
    t_composition = tuple(sorted(composition))

    # Check in the cache to see if PyxTal has previously been called to
    # validate this space group and composition
    if group_index in _pyxtal_check_compatible_cache:
        if t_composition in _pyxtal_check_compatible_cache[group_index]:
            return _pyxtal_check_compatible_cache[group_index][t_composition]
    else:
        _pyxtal_check_compatible_cache[group_index] = {}

    # Obtain the space group object
    space_group = get_space_group(group_index)

    # Perform compatibility check
    is_compatible = space_group.check_compatible(composition)[0]

    # Store result in cache before returning it
    _pyxtal_check_compatible_cache[group_index][t_composition] = is_compatible
    return is_compatible
