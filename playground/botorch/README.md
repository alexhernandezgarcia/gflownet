# Dataset
Hartmann Test Function: https://github.com/pytorch/botorch/blob/main/botorch/test_functions/synthetic.py#L305

# Posterior
 1. To create the posterior of the maximum of the black box function, $$F*$$ (Called only once when the class is initiliased)
 2. To create posterior of $$Y$$, outcome of experiment with no noise (Every forward call)
 3. Create posterior of $$Y$$, outocme of experiment with observation noise

# GP Implemenatation
Lewt m be number of candidate points, i.e, test data points
1. Posterior of F*
    a. Covar is non diagonal m x x
    b. Mean is of shape m
    c. mvn.batch_shape is []
    d. mvn.event_shape is [m]
2. Posterior of Y
    a. Covar is m x 1 x 1 x1
    b. Mean is m x 1 x 1
    c. mvn.batch_shape is 10 x 1
    d. mvn.event_shape is [1]



# Scripts
1. mes_gp.py: Test MES with Botorch GP module and posterior
2. mes_gp_debug.py: Instead of using the Botorch GP module and posterior, user-defined functions are used.
mes_nn_like_gp: Creates the covar matrix, and mean of the same dimensions as is in the GP implementation
3. mes_nn_bao_fix: The fix that Bao implemented to amke the code compatible with Botorch acquisition function.
mes_nn_like_gp_nondiagonal: Creates a non-diagonal convariance matrix for the posterior of F*
4. mes_nn_harcode_gpVal: Taking the covar and mean values from the GP run, and hardcoding them. COnclusion: Issue is not with the way mvn is constructed from the covar and mean. Issue is with the covar values
5. mes_exact_deepKernel: deep Kernel method using exact GP inference (only helpful with small dataset)
6. mes_var_deepKernel: deep Kernel with variational inference to scale GPs to larger datasets