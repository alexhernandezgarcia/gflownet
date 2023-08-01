import numpy
from pyxtal.symmetry import Group

_pyxtal_space_group_cache = {}
_pyxtal_check_compatible_cache = {}
_pyxtal_space_group_free_wp_multiplicity = {}
_pyxtal_space_group_wp_lowest_common_factor = {}


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
    # Remove from composition the number of atoms that are a multiple of the
    # space group's most specific free wyckoff position. This improves the
    # cache hit rate without affecting the validity of the composition
    free_multiplicity = space_group_lowest_free_wp_multiplicity(group_index)
    composition = [c for c in composition if c % free_multiplicity != 0]

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


def space_group_lowest_free_wp_multiplicity(group_index):
    """
    Returns the multiplicity of a space group's most specific free WP.

    This methods includes a lazy caching mechanism since the call to PyXtal
    methods to determine if a Wyckoff position is fixed or free is expensive.

    Args
    ----
    group_index : int
        Index (starting at 1) of the space group

    Returns
    -------
    multiplicity : int
        Multiplicity of the most specific free wyckoff position.
    """
    # Check in the cache to see if the multiplicity has previously been
    # computed for this space group
    if group_index in _pyxtal_space_group_free_wp_multiplicity:
        return _pyxtal_space_group_free_wp_multiplicity[group_index]

    # Obtain reference to the space group
    space_group = get_space_group(group_index)

    # Iterate over all of the space group's wyckoff positions from most
    # specific to most general until a wyckoff position with a degree of
    # freedom is found.
    multiplicity = None
    for wyckoff_idx in range(1, len(space_group.wyckoffs) + 1):
        wyckoff_position = space_group.get_wyckoff_position(-wyckoff_idx)
        if wyckoff_position.get_dof() > 0:
            multiplicity = wyckoff_position.multiplicity
            break

    # Store the result in cache before retuning it
    _pyxtal_space_group_free_wp_multiplicity[group_index] = multiplicity
    return multiplicity


def space_group_wyckoff_gcd(group_index):
    """
    Returns the greatest common divisor of a space group's Wyckoff positions

    This methods includes a lazy caching mechanism.

    Args
    ----
    group_index : int
        Index (starting at 1) of the space group

    Returns
    -------
    gcd : int
        Greatest common divisor of the group's wyckoff position.
    """
    # Check in the cache to see if the lowest common factor has previously
    # been computed for this space group
    if group_index in _pyxtal_space_group_wp_lowest_common_factor:
        return _pyxtal_space_group_wp_lowest_common_factor[group_index]

    # Obtain reference to the space group
    space_group = get_space_group(group_index)

    # Iterate over all of the space group's wyckoff positions from most
    # specific to most general until a wyckoff position with a degree of
    # freedom is found.
    multiplicities = []
    for wyckoff_idx in range(0, len(space_group.wyckoffs)):
        wyckoff_position = space_group.get_wyckoff_position(wyckoff_idx)
        multiplicities.append(wyckoff_position.multiplicity)
    gcd = numpy.gcd.reduce(multiplicities)

    # Store the result in cache before retuning it
    _pyxtal_space_group_wp_lowest_common_factor[group_index] = gcd
    return gcd
