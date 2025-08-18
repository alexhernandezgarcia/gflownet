"""
Base class for GFlowNet losses or objective functions.

.. warning::

    Should not be used directly, but subclassed to implement specific losses or
    objective functions for training a GFlowNet.
"""

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Union

from torch.nn import Parameter
from torchtyping import TensorType

from gflownet.policy.base import Policy
from gflownet.utils.batch import Batch
from gflownet.utils.common import set_device, set_float_precision


class BaseLoss(metaclass=ABCMeta):
    # TODO: study whether the dependence on the environment can be removed.
    # TODO: improve dependence on policies (needs re-implementation of policies)
    def __init__(
        self,
        env_maker: partial,
        forward_policy: Policy,
        backward_policy: Policy = None,
        state_flow: dict = None,
        logZ: Parameter = None,
        device: str = "cpu",
        float_precision: int = 32,
    ):
        """
        Base class for GFlowNet losses.

        Parameters
        ----------
        env_maker : partial
            The environment maker of the environments used to train the GFlowNet.
        forward_policy : :py:class:`gflownet.policy.base.Policy`
            The forward policy to be used for training. Parameterized from
            `gflownet.yaml:forward_policy` and parsed with
            `gflownet/utils/policy.py:set_policy`.
        bacward_policy : :py:class:`gflownet.policy.base.Policy`, optional
            Same as forward_policy, but for the backward policy.
        state_flow : dict, optional
            State flow config dictionary. See `gflownet.yaml:state_flow` for details. By
            default None.
        logZ : Parameter, optional
            The learnable parameters for the log-partition function logZ. By default
            None. It may be extended to consider modelling logZ with a neural network.
        device : str or torch.device
            The device to be passed to torch tensors.
        float_precision : int or torch.dtype
            The floating point precision to be passed to torch tensors.

        Attributes
        ----------
        env : Environment
            The environment used to train the GFlowNet.
        forward_policy : gflownet.policy.base.Policy
            The forward policy to be used for training. Parameterized from
            `gflownet.yaml:forward_policy` and parsed with
            `gflownet/utils/policy.py:set_policy`.
        backward_policy : gflownet.policy.base.Policy
            Same as forward_policy, but for the backward policy.
        state_flow : dict
            State flow config dictionary.
            default None.
        logZ : Parameter
            The learnable parameters for the log-partition function logZ.
        device : torch.device
            The device to be passed to torch tensors.
        float : torch.dtype
            The floating point precision to be passed to torch tensors.
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
        # Environment
        self.env = env_maker()
        # Policy models and parameters
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.state_flow = state_flow
        self.logZ - logZ
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Names, acronym and ID
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

        It is expected that one of the keys in the dictionary is 'all' and its value
        corresponds to the overall loss, which may be used to compute the gradient with
        respect to graph leaves with `backward()`.

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

    def compute(
        self, batch: Batch, get_sublosses: bool = False
    ) -> Union[float, dict[str, float]]:
        """
        Computes and aggregates the losses of a batch of states or trajectories.

        Parameters
        ----------
        batch : Batch
            A batch of states or trajectories.
        get_sublosses : bool
            Whether specific, relevant sub-aggregations of the loss should be computed
            and returned as a dictionary. Example of sub-losses are the average loss
            over the terminating states, over the intermediate states, over the
            on-policy trajectories, over the replay buffer trajectories, etc. If True,
            the returned variable is a dictionary. If False, simply the mean over all
            losses in the batch is returned.

        Returns
        -------
        float or dict
            A float containing the average loss or dictionary of loss aggregations,
            depending on the value of `get_sublosses`.
        """
        losses = self.compute_losses_of_batch(batch)
        if get_sublosses:
            return self.aggregate_losses_of_batch(losses, batch)
        else:
            return losses.mean()

    def set_log_z(self, logZ: Parameter):
        """
        Sets the input logZ as an attribute of the class instance.

        Parameters
        ----------
        logZ : Parameters
            The learnable parameters for the log-partition function logZ.
        """
        self.logZ = logZ
