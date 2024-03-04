# GFlowNet environments

This documentation is under construction.

## Option to potentially reduce the running time by skipping mask checks in `step()`

In certain environments, one of the most computationally demanding operations is the computation of the mask of invalid actions (the call to `get_masks_invalid_actions_*()`) . However, this computation is repeated in several parts of the code, so as to avoid sampling invalid actions, which could result in even worse computational efficiency and convergence rate . For example, the mask is computed when sampling the actions, and it is computed again in the `step()` method.

By default, this redundancy is in place. However, the user may choose to manually skip the computation of the masks in `step()` and therefore the check of whether an action is valid or invalid according to the mask before attempting to execute the step. There are two options:
- To skip the check in an entire run, set the environment configuration variable `skip_mask_check` to `True`.
- To skip the check in specific calls to `step()`, pass it the argument `skip_mask_check=True`.

Note that the mask of invalid actions indeed flags _invalid_ actions, as opposed to determining which actions are `valid`. In other words, actions flagged as invalid in the mask (`True`) must indeed be invalid, but actions marked as "not invalid" (`False`) are not necessarily valid. This may be the case if verifying the validity of action is computationally expensive, but determining a set of actions that are invalid may be affordable.

## Buffer, train data and test data

A train and a test set can be created at the beginning of training. The train set may be used to sample offline (backward) trajectories. The test set may be used to compute metrics during and after training. These sets may be created in different ways, specificied by the configuration variables `env.buffer.train.type` and `env.buffer.test.type`. Options for the data set `type` are

- `all`: all terminating states in the output space $\mathcal{X}$ will be added - Convenient but only feasible for small, synthetic environments like the hyper-grid.
- `grid`: a grid of points in the output space $\mathcal{X}$ - Only available in certain environments where obtaining a grid of points is meaningful. This mode also requires specifying the number of points via `env.buffer.<train/test>.n`.
- `uniform`: points sampled uniformly in the output space $\mathcal{X}$ - This mode also requires specifying the number of points via `env.buffer.<train/test>.n` and the random seed via `env.buffer.<train/test>.seed`.
- `random`: points sampled randomly (but not strictly uniformly) in the output space $\mathcal{X}$ - This mode also requires specifying the number of points via `env.buffer.<train/test>.n` and the random seed via `env.buffer.<train/test>.seed`.

Furthermore, a requirement to make use of a train or test set is to specify the name of the output CSV and pickle files, via `env.buffer.<train/test>.output_csv` and `env.buffer.<train/test>.output_pkl`.

An example of a buffer configuration with train (grid) and test (all) sets is the following:

```yaml
buffer:
  data_path: null
  train:
    type: grid
    n: 10
    output_csv: grid_train.csv
    output_pkl: grid_train.pkl
  test:
    type: all
    output_csv: grid_test.csv
    output_pkl: grid_test.pkl
```

The generation of a train and/or test set can be disabled by sampling their configuration to None/null (`env.buffer.train=null` via the command line).

### Replay buffer

It is possible to store the best trajectories, according to the reward, in a buffer and potentially use them for backward sampling or simply for evaluation during or at the end of training. In order to enable the replay buffer, its capacity must be larger than zero (default), which can be set with the configuration variable `env.buffer.replay_capacity`.

To use the replay buffer (once enabled) for backward sampling, one can specify `gflownet.optimizer.batch_size.backward_replay`. Currently, backward sampling is supported in one of two modes (specified using `gflownet.replay_sampling`):
- `permutation` (default): terminating states from the replay buffer are randomly permuted at each iteration, and first `gflownet.optimizer.batch_size.backward_replay` of them are used to sample trajectories,
- `weighted`: terminating states are sampled with probability equal to their reward.

:::{tip}
You can use [MyST](https://myst-parser.readthedocs.io/en/latest/syntax/admonitions.html) in the documentation. This is expected to fail on Github.
:::
