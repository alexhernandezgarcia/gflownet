from gflownet.losses.base import BaseLoss


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
        regularisers: list
            A list of instances of the regularisers
        """
        if not self.are_compatible(loss, regularisers):
            raise Exception(
                f"Loss {loss.id} and regularisers {[reg.id for reg in regularisers]} are not compatible"
            )

        self.loss = loss
        self.regularizers = regularisers

        self._requires_log_z = self.loss.requires_log_z

    @staticmethod
    def are_compatible(loss: BaseLoss, regularizers: List) -> bool:
        """
        Checks if loss and regularisers have the same aggregation strategy

        Parameters
        ----------
        loss : BaseLoss or its child
            An instance of the loss function
        regularisers: list
            A list of instances of the regularisers

        Returns
        -------
        bool
            True if they all have the same aggregation strategy
        """
        for reg in regularisers:
            if reg.aggregates_over() != loss.aggregates_over():
                return False
        return True

    def aggregates_over(self) -> str:
        """
        Returns a label indentifying over which objects in the batch
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
            reg.requires_backward_policy() for reg in self.regularisers
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
            reg.requires_state_flow_model() for reg in self.regularisers
        ]
        return any(flags)

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function and all the regularisers are
        well defined for continuous GFlowNets, that is continuous environments,
        or False otherwise.

        Returns
        -------
        bool
        """
        flags = [self.loss.is_defined_for_continuous()] + [
            reg.is_defined_for_continuous() for reg in self.regularisers
        ]
        return all(flags)

    def compute_losses_of_batch(self, batch: Batch) -> Dict[TensorType["batch_size"]]:
        """
        Computes loss and regularisers for each trajectory or state
        of the input batch.

        Parameters
        ----------
        batch : Batch
            A batch of trajectories or states.

        Returns
        -------
        dict
            The loss and regularisers of each trajectory / state in the batch.
        """
        loss_batch = self.loss.compute_losses_of_batch(batch)
        regularisers_batch = [
            reg.compute_losses_of_batch(batch) for reg in self.regularisers
        ]
        result = {"loss": loss_batch, "regularizers": regularisers_batch}

    def aggregate_losses_of_batch(
        self, losses: Dict[TensorType["batch_size"]], batch: Batch
    ) -> dict[str, float]:
        """
        Aggregates the losses and regularisers computed from a batch
        to obtain the overall average loss.

        Returns values of the overall loss with and w/o regularisers
        as well as overall values of the regularisers themselves.

        The result is returned as a dictionary with the following items:
        - 'all': overall average loss with regularisers.
        - 'all no regularizers': overall average loss w/o regularisers.
        -

        Parameters
        ----------
        losses : dict of tensors
            Dictionary with the loss  and regularisers of each
            trajectory/state in the batch
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        loss_dict : dict
            A dictionary of loss aggregations.
        """
        all_tensors = torch.stack([losses["loss"]] + losses["regularizers"], dim=0)
        loss_reg = torch.sum(all_tensors, dim=0)

        # Compute loss w/ and w/o regularisers
        loss_reg = self.loss.aggregate_losses_of_batch(loss_reg, batch)
        loss = self.loss.aggregate_losses_of_batch(losses["loss"], batch)
        regs = [
            reg.aggregate_losses_of_batch(reg_batch, batch, self.loss)
            for reg, reg_batch in zip(self.regularisers, losses["regularizers"])
        ]

        # Combine results in one dict
        for key, value in loss.items():
            new_key = f"{key} no regularizers"
            loss_reg[new_key] = value
        for reg_dict in regs:
            for key, value in reg_dict:
                assert keyy not in loss_reg.keys()
                loss_reg[key] = value

        return loss_reg
