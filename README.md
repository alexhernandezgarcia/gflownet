# GFlowNet

this repository implements GFlowNets, generative flow networks for probabilistic modelling, on PyTorch. A design guideline behind this implementation is the separation of the logic of the GFlowNet agent and the environments on which the agent can be trained on. In other words, this implementation should allow its extension with new environments without major or any changes to to the agent. Another design guideline is flexibility and modularity. The configuration is handled via the use of [Hydra](https://hydra.cc/docs/intro/).

## Installation

### pip

```
python -m pip install --upgrade https://github.com/alexhernandezgarcia/example_pypi_package/archive/main.zip
```
