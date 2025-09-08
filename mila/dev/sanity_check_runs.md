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
python train.py user=$USER env=grid env.length=10 proxy=box/corners gflownet=trajectorybalance loss=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=all
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=grid env.length=10 proxy=box/corners gflownet=trajectorybalance loss=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=all
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=grid env.length=10 proxy=box/corners gflownet=trajectorybalance loss=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=all
```

### Flow Matching loss

`salloc`:

```bash
python train.py user=$USER env=grid env.length=10 proxy=box/corners gflownet=flowmatch loss=flowmatching device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=all
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=grid env.length=10 proxy=box/corners gflownet=flowmatch loss=flowmatching device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=all
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=grid env.length=10 proxy=box/corners gflownet=flowmatch loss=flowmatching device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=all
```

## Tetris

- Width 5
- Height 10

### Trajectory Balance loss

`salloc`:

```bash
python train.py user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=trajectorybalance loss=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100 buffer.test.type=random buffer.test.n=10
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=trajectorybalance loss=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100 buffer.test.type=random buffer.test.n=10
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=trajectorybalance loss=trajectorybalance device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100 buffer.test.type=random buffer.test.n=10
```

### Flow Matching loss

`salloc`:

```bash
python train.py user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=flowmatch loss=flowmatching device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100 buffer.test.type=random buffer.test.n=10
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=flowmatch loss=flowmatching device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100 buffer.test.type=random buffer.test.n=10
```
losssbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER env=tetris proxy=tetris env.width=5 env.height=10 gflownet=flowmatch loss=flowmatching device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True evaluator.top_k=10 evaluator.n_top_k=100 buffer.test.type=random buffer.test.n=10
```

## Continuous Torus as in Lahlou et al (ICML 2023)

- [Paper](https://arxiv.org/abs/2301.12594)
- config: `experiments=icml23/ctorus`

`salloc`:

```bash
python train.py user=$USER +experiments=icml23/ctorus evaluator.period=500 device=cpu logger.project_name=gfn_sanity_checks loss=trajectorybalance logger.do.online=True buffer.test.type=grid buffer.test.n=1000
```

`sbatch` with `virtualenv`:

```bash
python mila/launch.py --venv=<path-to-env> --template=mila/sbatch/template-venv.sh user=$USER +experiments=icml23/ctorus loss=trajectorybalance evaluator.period=500 device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=grid buffer.test.n=1000
```

`sbatch` with `conda`:

```bash
python mila/launch.py --conda_env=<conda-env-name> user=$USER +experiments=icml23/ctorus loss=trajectorybalance evaluator.period=500 device=cpu logger.project_name=gfn_sanity_checks logger.do.online=True buffer.test.type=grid buffer.test.n=1000
```
