# Dataset
Hartmann Test Function: https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py#L305

# Posterior
 1. To create the posterior of the maximum of the black box function, $$F*$$ (Called only once when the class is initiliased)
 2. To create posterior of $$Y$$, outcome of experiment with no noise (Every forward call)
 3. Create posterior of $$Y$$, outocme of experiment with observation noise

# GP Implemenatation
Let m be number of candidate points, i.e, test data points
1. Posterior of F* \\
    a. Covar is non diagonal m x x
    b. Mean is of shape m
    c. mvn.batch_shape is []
    d. mvn.event_shape is [m]
2. Posterior of Y
    a. Covar is m x 1 x 1 x1
    b. Mean is m x 1 x 1
    c. mvn.batch_shape is 10 x 1
    d. mvn.event_shape is [1]


# Where can the error be?
1. Posterior for $$F*$$: 
2. Posterior for $$Y$$: \textit{I made the shapes match and the kind of matrix (diag vs non-diag. In the GP implementation, there is a different covar matrix for each test point. That makes sense from the NN point of view as well. So I have a feeling the issue is not here.}
3. We don't modify our data points while considering `observation_noise = True`. {I tried the same for GPs, ie, did not modify the data points even when `observation_noise=True` and GP gives postive MES values means that the `posterior()` should work even if `observation_noise=True` doesn't lead to a different treatment. So I am confident the issue is not here.}

# What I am not clear with? 
In the GP implementation, the covar_matrix for $$F*$$ is a one covariance matrix for the entire batch of m test points. The covar matrix for $$Y$$ is a batch of m covariance matrices -- one for each test point, When the ultimate goal is to sample from these posteriors, why are they constructed in different ways? I need to dig deeper into the maths to understand this


# Scripts
1. `mes_gp.py`: Test MES with Botorch GP module and posterior
2. `mes_gp_debug.py`: Instead of using the Botorch GP module and posterior, user-defined functions are used.
3. `mes_nn_like_gp`: Creates the covar matrix, and mean of the same dimensions as is in the GP implementation
4. `mes_nn_bao_fix`: The fix that Bao implemented to amke the code compatible with Botorch acquisition function.
5. `mes_nn_like_gp_nondiagonal`: Creates a non-diagonal convariance matrix for the posterior of F*
6. `mes_nn_hardcode_gpVal`: Taking the covar and mean values from the GP run, and hardcoding them. Conclusion: Issue is not with the way mvn is constructed from the covar and mean. Issue is with the covar values
7. `mes_exact_deepKernel`: deep Kernel method using exact GP inference (only helpful with small dataset)
8. `mes_var_deepKernel`: deep Kernel with variational inference to scale GPs to larger datasets

# Tutorials
1. https://botorch.org/tutorials/max_value_entropy