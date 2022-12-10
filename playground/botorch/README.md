# Setting

## Dataset
[Hartmann Test Function](https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py#L305)

`trainX` is a batch of `1x6` random vectors, and `trainY` is the corresponding `Hartmann()` value.

## Posterior()
The posterior function is called multiple times in different parts of the qMES source code.
 1. To create the posterior of the maximum of the black box function, $F*$ (Called only once when the class is initialised) [[Source]](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/max_value_entropy_search.py#L918)
 2. To create posterior of $Y$, outcome of experiment with `observation_noise=False` [[Source]](https://github.com/pytorch/botorch/blob/041125fe3bd022693b90f157d675f492463f67ce/botorch/acquisition/max_value_entropy_search.py#L127)

 3. Create posterior of $Y$, outcome of experiment with `observation noise = True` 
 2 and 3 are called with each `forward()` [[Source]](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/max_value_entropy_search.py#L428)

# GP Implementation
Let $m$ be number of candidate points, i.e, test data points
1. Posterior of $F*$ 
    1. Covar is non diagonal `m x m`
    2. Mean is of shape `m`
    3. mvn.batch_shape is `[]`
    4. mvn.event_shape is `[m]`
2. Posterior of $Y$
    1. Covar is `m x 1 x 1 x1`
    2. Mean is `m x 1 x 1`
    3. mvn.batch_shape is `10 x 1`
    4. mvn.event_shape is `[1]`

# Remarks
## Testing the Acqusition Function
The only check performed right now is that of strictly positive acqusition function values. For this, I create $n$ batches of test points, each with batch_size $b$, feed it to the acquisition function. I print the MES values of the batch if MES value corresponding to even one entry of the batch is negative.


## What is the psoterior used for?

### Posterior of $F*$
`posterior.mean` and `posterior.variance` used in `sample_max_gumbel()`

#### Posterior of $Y$
1. `compute_information_gain()`:
    1. `posterior.mean`
    2. `posterior.mvn.covariance_matrix`
    3. `posterior.mvn.entropy()` calls `torch.distribution.multivariate_normal.entropy()` [Source](https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal.entropy) as there is no `gpytorch.distributions.multivariate_normal.entropy()` implementation
    4. `posterior.mvn.log_prob`: [Source](https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/distributions/multivariate_normal.py#L165)
2. `get_posterior_sampler()` [Source](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/max_value_entropy_search.py#L440)
    1. `posterior.base_sample_shape`: [Source](https://github.com/pytorch/botorch/blob/main/botorch/posteriors/gpytorch.py#L71)
    2. `posterior.rsample`: [Source](https://github.com/pytorch/botorch/blob/main/botorch/posteriors/gpytorch.py#L140)

## Where can the error be?
1. Posterior for $F*$: 
2. Posterior for $Y$: *I made the shapes match and the kind of matrix (diag vs non-diag.).So I have a feeling the issue is not here. In the GP implementation, there is a different covar matrix for each test point. That makes sense from the NN point of view as well.*
3. My implemenatation for the `posterior()` was super-basic. It did not modify  data points when `observation_noise = True`. {I tried the same for GPs, ie, did not modify the data points even when `observation_noise=True` and GP still gives postive MES values. This means that the `posterior()` should work even if `observation_noise=True` doesn't lead to a different treatment. So I feel the issue is not here.}

## What I am not clear with? 
In the GP implementation, the covar_matrix for $F*$ is one covariance matrix for the entire batch of m test points. The covar matrix for $Y$ is a batch of m covariance matrices -- one for each test point, When the ultimate goal is to sample from these posteriors, why are they constructed in different ways? I need to dig deeper into the maths to understand this

# Observations
I ran a couple of random experiments with the NN-based proxy. When I say *hardcoded*, it implies I took the value given by the GP and hardcoded it in the NN script, instead of computing it on-the-fly from the data.

### Experiment 1
Mean and Covariance for $F*$: hard-coded
Mean and Covariance for $Y$: calculated from the data
Result: Negative MES Values

### Experiment 2
Mean and Covariance for $F*$: calculated from the data
Mean and Covariance for $Y$: hard-coded
Result: Positive MES Values

This result was irrpespective of whether I force the covariance matrix to be diagonal or non-diagonal. 

My hunch is that since we just use the `mean` and `variance` (which are  pretty straightforward to construct) from the posterior of $F*$, the issue should be in the posterior of $Y$ instead.

### 

# Scripts
1. `mes_gp.py`: Test MES with Botorch GP module and posterior
2. `mes_gp_debug.py`: Instead of using the Botorch GP module and posterior, user-defined class for GP is used.
3. `mes_nn_like_gp.py`: Creates the covar matrix, and mean of the same dimensions as is in the GP implementation. The values do not match -- covar is diagonal for posterior of $F*$
4. `mes_nn_bao_fix.py`: The fix that Bao implemented to make the code compatible with Botorch acquisition function.
5. `mes_nn_like_gp_nondiagonal.py`: Creates a non-diagonal convariance matrix for the posterior of F*
6. `mes_nn_hardcode_gpVal.py`: Taking the covar and mean values from the GP run, and hardcoding them in this script. Conclusion: Issue is not with the way mvn is constructed from the covar and mean. Issue is with the covar values.
7. `mes_exact_deepKernel.py`: deep Kernel method using exact GP inference (only helpful with small dataset)
8. `mes_var_deepKernel.py`: deep Kernel with variational inference to scale GPs to larger datasets
9. `mes_nn_explicit_batch_1.py`: In 2 and 3, `mvn.batch_shape=[]` . Here I explcitiy forced it to be 1 by unsqueezing the covariance matrix before feeding it to `mvn`
10. `mes_nn_hardcode2_gpVal.py`: Playground script that Alex and I were working on -- were comparing the posterior given by the GP vs. NN

# References
1. [MaxValueEntropy Tutorial kind by Botorch](https://botorch.org/tutorials/max_value_entropy): Derives the formula.
2. [Gpytorch Multivariate Normal Docs](https://docs.gpytorch.ai/en/stable/distributions.html#multivariatenormal)
3. [Gpytorch Multivariate Normal Source Code](https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/distributions/multivariate_normal.py)
4. [Exact GP Regression Module Tutorial](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html): How to define a GP Model
5. [Abstract Class Implementation of Botrch Mode](https://github.com/pytorch/botorch/blob/main/botorch/models/model.py#L46): Abstract method `posterior()` that needs to be implemented.
6. [qMaxValueEntropy Source Code](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/max_value_entropy_search.py#L300)
8. [`sample_gumbel_values()`](https://github.com/pytorch/botorch/blob/main/botorch/acquisition/max_value_entropy_search.py#L893): Source Code to sample from posterior of $F*$ to create `max_samples`
9. [MFBO MES Paper](https://arxiv.org/pdf/1901.08275.pdf): Appendix C has the mathematical derivation for MES 
10. [Source Code for Botorch Posteriors](https://botorch.org/api/_modules/botorch/posteriors/gpytorch.html)