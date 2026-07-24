from typing import List, Union

import torch
from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch


class RegularizedLoss(BaseLoss):
    """
    A wrapper class allowing to combine any loss with regularizers.
    """

    def __init__(self, loss: BaseLoss, regularizers: List):
        """
        Initialization method for the RegularizedLoss class.

        Parameters
        ----------
        loss : BaseLoss or its child
            An instance of the loss function
        regularizers : list
            A list of instances of the regularizers
        """
        if not self.are_compatible(loss, regularizers):
            raise Exception(
                f"Loss {loss.id} and regularizers "
                f"{[reg.id for reg in regularizers]} are not compatible"
            )

        self.loss = loss
        self.regularizers = regularizers

        self._requires_log_z = self.loss.requires_log_z or any(
            reg.requires_log_z for reg in self.regularizers
        )

    @staticmethod
    def are_compatible(loss: BaseLoss, regularizers: List) -> bool:
        """
        Checks if loss and regularizers have the same aggregation strategy

        Parameters
        ----------
        loss : BaseLoss or its child
            An instance of the loss function
        regularizers : list
            A list of instances of the regularizers

        Returns
        -------
        bool
            True if they all have the same aggregation strategy
        """
        for reg in regularizers:
            if reg.aggregates_over() != loss.aggregates_over():
                return False
        return True

    def aggregates_over(self) -> str:
        """
        Returns a label identifying over which objects in the batch
        aggregation happens.

        Returns
        -------
        str
            Aggregation strategy of the loss
        """
        return self.loss.aggregates_over()

    def requires_backward_policy(self) -> bool:
        """
        Returns True if the loss function or any of the regularizers
        require a backward policy.

        Returns
        -------
        bool
        """
        flags = [self.loss.requires_backward_policy()] + [
            reg.requires_backward_policy() for reg in self.regularizers
        ]
        return any(flags)

    def requires_state_flow_model(self) -> bool:
        """
        Returns True if the loss function or any of the regularizers
        require a state flow model.

        Returns
        -------
        bool
        """
        flags = [self.loss.requires_state_flow_model()] + [
            reg.requires_state_flow_model() for reg in self.regularizers
        ]
        return any(flags)

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function and all the regularizers are
        well defined for continuous GFlowNets, that is continuous environments,
        or False otherwise.

        Returns
        -------
        bool
        """
        flags = [self.loss.is_defined_for_continuous()] + [
            reg.is_defined_for_continuous() for reg in self.regularizers
        ]
        return all(flags)

    def compute_losses_of_batch(
        self, batch: Batch
    ) -> dict[str, Union[TensorType["batch_size"], List[TensorType["batch_size"]]]]:
        """
        Computes loss and regularizers for each trajectory or state
        of the input batch.

        Parameters
        ----------
        batch : Batch
            A batch of trajectories or states.

        Returns
        -------
        dict
            The loss and regularizers of each trajectory / state in the batch.
        """
        loss_batch = self.loss.compute_losses_of_batch(batch)
        regularizers_batch = [
            reg.compute_losses_of_batch(batch) for reg in self.regularizers
        ]
        result = {"loss": loss_batch, "regularizers": regularizers_batch}
        return result

    def aggregate_losses_of_batch(
        self, losses: dict[str, TensorType["batch_size"]], batch: Batch
    ) -> dict[str, float]:
        """
        Aggregates the losses and regularizers computed from a batch
        to obtain the overall average loss.

        Returns values of the overall loss with and w/o regularizers
        as well as overall values of the regularizers themselves.

        The result is returned as a dictionary with the following items:
        - 'all': overall average loss with regularizers.
        - 'all no regularizers': overall average loss w/o regularizers.

        Parameters
        ----------
        losses : dict of tensors
            Dictionary with the loss and regularizers of each
            trajectory/state in the batch
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        loss_dict : dict
            A dictionary of loss aggregations.
        """
        all_tensors = torch.stack([losses["loss"]] + losses["regularizers"], dim=0)
        combined = torch.sum(all_tensors, dim=0)

        # Compute loss w/ and w/o regularizers
        loss_reg = self.loss.aggregate_losses_of_batch(combined, batch)
        loss = self.loss.aggregate_losses_of_batch(losses["loss"], batch)
        regs = [
            reg.aggregate_losses_of_batch(reg_batch, batch, self.loss)
            for reg, reg_batch in zip(self.regularizers, losses["regularizers"])
        ]

        # Combine results in one dict
        for key, value in loss.items():
            new_key = f"{key} no regularizers"
            loss_reg[new_key] = value
        for reg_dict in regs:
            for key, value in reg_dict.items():
                assert key not in loss_reg.keys()
                loss_reg[key] = value

        return loss_reg
