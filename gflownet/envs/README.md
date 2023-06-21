# GFlowNet environments

This documentation is under construction.

## Option to potentially reduce the running time by skipping mask checks in `step()`

In certain environments, one of the most computationally demanding operations is the computation of the mask of invalid actions (the call to `get_masks_invalid_actions_*()`) . However, this computation is repeated in several parts of the code, so as to avoid sampling invalid actions, which could result in even worse computational efficiency and convergence rate . For example, the mask is computed when sampling the actions, and it is computed again in the `step()` method.

By default, this redundancy is in place. However, the user may choose to manually skip the computation of the masks in `step()` and therefore the check of whether an action is valid or invalid according to the mask before attempting to execute the step. There are two options:
- To skip the check in an entire run, set the environment configuration variable `skip_mask_check` to `True`.
- To skip the check in specific calls to `step()`, pass it the argument `skip_mask_check=True`.
