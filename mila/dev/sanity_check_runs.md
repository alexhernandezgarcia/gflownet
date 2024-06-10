# Dev Sanity Check Runs

Here is a list of commands to train GFlowNets on various on environments with different configurations that can be used as sanity checks during development of the repository. A number of such runs can be found in [alexhg's `gfn_sanity_checks` wandb project](https://wandb.ai/alexhg/gfn_sanity_checks). Unless fundamental things about the training process change, it can be expected that the training curves of, for instance, the Loss, `mean_rewards`, logZ and Jensen Shannon Div (if available) are very similar if not identical across runs.

In order to launch all the sanity checks as individual jobs:

`venv`:

```bash
python mila/launch.py --venv=<path-to-venv> --template=mila/sbatch/template-venv.sh --jobs=mila/dev/sanity_check_runs.yaml
```

`conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> --jobs=mila/dev/sanity_check_runs.yaml
```

**Note**: the default modules loaded are `cuda/11.3` and `anaconda/3`. Add `--modules='module1 module2 <etc.>'` to specify your own modules.

## Grid

- 2 dimensions
- Length 10

### Trajectory Balance loss

`salloc`:

```bash
python main.py user=$USER env=grid env.length=10 gflownet=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=grid env.length=10 gflownet=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=grid env.length=10 gflownet=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

### Flow Matching loss

`salloc`:

```bash
python main.py user=$USER env=grid env.length=10 gflownet=flowmatch device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=grid env.length=10 gflownet=flowmatch device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=grid env.length=10 gflownet=flowmatch device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

## Tetris

- Width 5
- Height 10

### Trajectory Balance loss

`salloc`:

```bash
python main.py user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100
```

### Flow Matching loss

`salloc`:

```bash
python main.py user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=flowmatch device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=flowmatch device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=flowmatch device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100
```

## Continuous Torus as in Lahlou et al (ICML 2023)

- [Paper](https://arxiv.org/abs/2301.12594)
- config: `experiments=icml23/ctorus`

`salloc`:

```bash
python main.py user=$USER +experiments=icml23/ctorus evaluator.period=500 device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER +experiments=icml23/ctorus evaluator.period=500 device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER +experiments=icml23/ctorus evaluator.period=500 device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True
```
