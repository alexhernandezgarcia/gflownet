"""
Base class for GFlowNet losses or objective functions.

.. warning::

    Should not be used directly, but subclassed to implement specific losses or
    objective functions for training a GFlowNet.
"""

import os
from abc import ABCMeta, abstractmethod

from torchtyping import TensorType

from gflownet.utils.batch import Batch


class BaseLoss(metaclass=ABCMeta):
    def __init__(self):
        """
        Base class for GFlowNet losses.

        Attributes
        ----------
        name : str
            The name of the loss or objective function. This is meant to be nicely
            formatted for printing purposes, for example using capital letters and
            spaces.
        acronym : str
            The acronym of the loss or objective function.
        id : str
            The identifier of the loss or objective function. This is for processing
            purposes.
        """
        self.name = "Base Loss"
        self.acronym = ""
        self.id = "base"

    @abstractmethod
    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function is well defined for continuous GFlowNets,
        that is continuous environments, or False otherwise.

        Returns
        -------
        bool
            Whether the loss function is well defined for continuous GFlowNets.
        """
        pass

    @abstractmethod
    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the loss for each state or trajectory of the input batch.

        Parameters
        ----------
        batch : Batch
            A batch of states or trajectories.

        Returns
        -------
        losses : tensor
            The loss of each unit in the batch. Depending on the loss function, the
            unit may be states (for example, for the flow matching loss) or
            trajectories (for example, for the trajectory balance loss). That is, the
            notion of batch size is different for each loss function, as it may refer
            to the number of states or the number of trajectories.
        """
        pass

    @abstractmethod
    def aggregate_losses_of_batch(
        self, losses: TensorType["batch_size"], batch: Batch
    ) -> dict[str, float]:
        """
        Aggregates the losses computed from a batch to obtain the average loss and/or
        multiple averages over different relevant parts of the batch.

        The result is returned as a dictionary whose keys are the identifiers of each
        type of aggregation and the values are the aggregated losses.

        Parameters
        ----------
        losses : tensor
            The loss of each unit (state or trajectory) in the batch.
        batch : Batch
            A batch of states or trajectories.

        Returns
        -------
        loss_dict : dict
            A dictionary of loss aggregations. The keys are the identifiers of each
            type of aggregation and the values are the aggregated losses.
        """
        pass
