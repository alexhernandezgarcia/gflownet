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
        """

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
