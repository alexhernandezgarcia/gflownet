"""

This submodule allows you to log additional data to display in a dashboard after
training. You can
    - explore subgraphs of the DAG and track changes in the logprobabilities of
    edges
    - Explore the state space and compare it to a dataset to how the model handles
    different areas, explore the correlation PF - Rewards in detail
In contrast to WandB the focus here lies in exploring the model in depth.

Main repo:
https://github.com/florianholeczek/GFlowNet_Training_Vis_Pilot

Requirements:
You will need four functions in your environment. All of them are implemented as default
in the base and stack environment but might need adaption for your env:

    - vis_states2texts (and possibly a return conversion texts2states):
        Should take a list of states in environment format and convert it to a list
        of unique string representations.
        This is needed to correctly identify a state in the database.
        The dashboard will use '#' as the default source state.
        By default repr(state) will be used to get a string representation.

    - vis_states2features:
        Takes a list of states in environment format and converts it to
            1. features
                np.Ndarray or torch Tensor with features of size
                (len(states), n_features)
            2. features_valid
                bool array or tensor of size (len(states),)
                indicating if features are valid
        These will be used in a dimensionality reduction to show the state space.
        By default state2policy will be used.

    - vis_show_state:
        Takes a state in the text format saved in the database.
        Should return either a string with a base64 encoded svg image or a list of
        strings to identify a state from.
        This is needed to display a state. If an image is possible it should be
        preferred. Examples:
             Molecule Env: drawing a molecule via rdkit.Chem.Draw.MolsToGridImage()
             Grid Env: Displaying the state in the grid
             Crystals: Plot the atoms in the unit cell
        If this is not possible (or as fallback for incomplete or invalid states) a list
        of strings can be returned. In this case these will be displayed as text.
        Try to keep the texts short to keep them readable.
        If you want to return only one string wrap it in a list otherwise it will be
        read as a base64 image.
        By default state2readable will be wrapped in a list and returned.

    -vis_aggregation:
        Takes a list of states in the text format saved in the database and returns
        either a string with a base64 encoded svg image or a list of strings.
        Works as vis_show_state but for multiple states.
        Should display something all states have in common.
        Examples:
            Molecule Env (Or similar graph based envs): Maximum common substructure
            Grid Env: All cells plottet in the grid
        The goal is that states that are similar one can identify a part of the state
        space  these states come from. As in vis_show_state you can also return a list
        of strings describing the common features of the states.
        This would be neccessary in the crystal env, where common point-groups,
        common compositions and the range of the lattice parameters can be shown.
        By default None is returned.

In the config files there is a vislogger configuration under the logger.
Enable it by setting vislogger=True in the basic logger and all relevant data will be
logged. For now the training samples will be logged, an option to sample new samples
on-policy might be added.


Configuration:
    - log_every_n
        The whole batch will be logged every n iterations.
    - use_env_feature_fn
        Keep true for now, another option to add features might be added later
    - show_during_training
        To display the dashboard during training and see live progress.
        Not implemented yet!
    - launch_after_training
        To automatically start the dashboard once training completed.

Note that the full trajectories will be stored. The database will have
n_samples * (n_iterations / log_every_n_iterations) * average trajectory length
entries. Keeping this below 1e6 should work fine.

You can also start the dashboard for a logged training run with the script in
gflownet/utils/vislogger/run_dashboard.py
See its documentation for how to do so.

Click 'How to use' in the upper left of the Dashboard to see information on how to use
and interpret it.

You can also add testdata to compare your samples with via the add_testset() function
in gflownet/utils/vislogger/add_testset.py
Rewards will be computed with the same proxy as in the training run.
Example:
    from gflownet.utils.vislogger.add_testset import add_testset

    logdir = 'path/to/your/logs/YYYY-MM-DD-.../visdata'
    states = ... # create the states from the dataset in environment format
    add_testset(logdir, states)

This can also be done iteratively for batches of states.

"""
