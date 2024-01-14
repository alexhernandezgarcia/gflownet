import json
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.crystals.clattice_parameters import CLatticeParameters
from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.envs.stack import Stack
from gflownet.utils.common import copy, tbool, tfloat, tlong
from gflownet.utils.crystals.constants import TRICLINIC


class CCrystal(Stack):
    """
    A combination of Composition, SpaceGroup and CLatticeParameters into a single
    environment. Works sequentially, by first filling in the Composition, then
    SpaceGroup, and finally CLatticeParameters.
    """

    def __init__(
        self,
        composition_kwargs: Optional[Dict] = None,
        space_group_kwargs: Optional[Dict] = None,
        lattice_parameters_kwargs: Optional[Dict] = None,
        do_composition_to_sg_constraints: bool = True,
        do_sg_to_composition_constraints: bool = True,
        do_sg_to_lp_constraints: bool = True,
        do_sg_before_composition: bool = False,
        **kwargs,
    ):
        self.do_sg_to_composition_constraints = (
            do_sg_to_composition_constraints and do_sg_before_composition
        )
        self.do_composition_to_sg_constraints = (
            do_composition_to_sg_constraints and not do_sg_before_composition
        )
        self.do_sg_to_lp_constraints = do_sg_to_lp_constraints
        self.do_sg_before_composition = do_sg_before_composition

        self.composition_kwargs = dict(
            composition_kwargs or {},
            do_spacegroup_check=self.do_sg_to_composition_constraints,
        )
        self.space_group_kwargs = space_group_kwargs or {}
        self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}

        composition = Composition(**self.composition_kwargs)
        space_group = SpaceGroup(**self.space_group_kwargs)
        # We initialize lattice parameters with triclinic lattice system as it is the
        # most general one, but it will have to be reinitialized using proper lattice
        # system from space group once that is determined.
        lattice_parameters = CLatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )

        # Initialize tuple of subenvs and stage indices
        if self.do_sg_before_composition:
            subenvs = (space_group, composition, lattice_parameters)
            self.stage_spacegroup = 0
            self.stage_composition = 1
        else:
            subenvs = (composition, space_group, lattice_parameters)
            self.stage_composition = 0
            self.stage_spacegroup = 1
        self.stage_latticeparameters = 2

        # Initialize base Stack environment
        super().__init__(subenvs=subenvs, **kwargs)

    def _apply_constraints(self, action: Tuple = None):
        """
        Applies constraints across sub-environments, when applicable.

        - composition -> space group (if composition is first)
        - space group -> composition (if space group is first)
        - space group -> lattice parameters

        This method is used in step() and set_state().
        """
        # Apply constraints composition -> space group
        # Apply constraint only if action is None or if it is the composition EOS
        if (
            self.subenvs[self.stage_composition].done
            and self.do_composition_to_sg_constraints
            and not self.do_sg_before_composition
            and (
                action is None
                or self._depad_action(action)
                == self.subenvs[self.stage_composition].eos
            )
        ):
            self.subenvs[self.stage_spacegroup].set_n_atoms_compatibility_dict(
                self.subenvs[self.stage_composition].state
            )

        # Apply constraints:
        # - space group -> composition
        # - space group -> lattice parameters
        # Apply constraint only if action is None or if it is the space group EOS
        if self.subenvs[self.stage_spacegroup].done and (
            action is None
            or self._depad_action(action) == self.subenvs[self.stage_spacegroup].eos
        ):
            if self.do_sg_before_composition and self.do_sg_to_composition_constraints:
                space_group = self.subenvs[self.stage_spacegroup].space_group
                self.subenvs[self.stage_composition].space_group = space_group
            if self.do_sg_to_lp_constraints:
                lattice_system = self.subenvs[self.stage_spacegroup].lattice_system
                self.subenvs[self.stage_latticeparameters].set_lattice_system(
                    lattice_system
                )

    # TODO: this could eventually be moved to Stack
    def process_data_set(self, data: List[List]) -> List[List]:
        is_valid_list = []
        for x in data:
            is_valid_list.append(
                all(
                    [
                        subenv.is_valid(self._get_substate(x, stage))
                        for stage, subenv in self.subenvs.items()
                    ]
                )
            )
        return [x for x, is_valid in zip(data, is_valid_list) if is_valid]
