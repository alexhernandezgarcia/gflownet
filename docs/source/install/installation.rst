Installation
============

This section provides detailed instructions on how to install **gflownet**. Ensure your system meets the following requirements before proceeding with the installation.

Requirements
------------

* **Python Version:** Python 3.10 is required.
* **CUDA Version:** CUDA 11.8 is needed for GPU acceleration (optional for CPU-only installation).
* **Operating System:** The setup is primarily supported on Ubuntu. It should also work on OSX, but you might need to handle the package dependencies manually.

Quick Installation
------------------

**If you simply want to install everything on a GPU-enabled machine, clone the repo and run** ``install.sh``:

.. code-block:: bash

    git clone git@github.com:alexhernandezgarcia/gflownet.git
    cd gflownet
    source install.sh

.. note::
    
    - This project **requires** Python 3.10 and CUDA 11.8.
    - It is also **possible to install a CPU-only environment** that supports most features (see below).
    - Setup is currently only supported on Ubuntu. It should also work on OSX, but you will need to handle the package dependencies.

Step by Step Installation
-------------------------

The following steps, as well as the script ``install.sh``, assume the use of Python virtual environments for the installation.

1. **Load Required Modules (if on a cluster):**

   Ensure that you have Python 3.10 and, if you want to install GPU-enabled PyTorch, CUDA 11.8. In a cluster that uses `modules <https://hpc-wiki.info/hpc/Modules>`_, you may be able to load Python and CUDA with:

   .. code-block:: bash

       module load python/3.10
       module load cuda/11.8

2. **Create and Activate Virtual Environment:**

   Create and activate a Python virtual environment with ``venv``. For example:

   .. code-block:: bash

       python -m venv gflownet-env
       source gflownet-env/bin/activate

3. **Install PyTorch 2.5.1:**

   For a **CUDA-enabled installation**:

   .. code-block:: bash

       python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
       python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

   For a **CPU-only installation**:

   .. code-block:: bash

       python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
       python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

4. **Install the Rest of the Dependencies:**

   .. code-block:: bash

       python -m pip install .

   The above command will install the minimum set of dependencies to run the core features of the gflownet package. Specific features require the installation of extra dependencies.

Optional Packages (Extras)
---------------------------

Currently, these are the existing sets of extras:

* **dev**: dependencies for development, such as linting and testing packages.
* **materials**: dependencies for materials applications, such as the Crystal-GFN.
* **molecules**: dependencies for molecular modelling and generation, such the Conformer-GFN.

Extras can be installed by specifying the tags in square brackets:

.. code-block:: bash

    python -m pip install .[dev]

or

.. code-block:: bash

    python -m pip install .[dev,materials]

Installing with ``install.sh``
------------------------------

The script ``install.sh`` simplifies the installation of a Python environment with the necessary or desired dependencies.

By default, running ``source install.sh`` will create a Python environment in ``./gflownet-env`` with CUDA-enabled PyTorch and all the dependencies (all extras). However, the script admits the following arguments to modify the configuration of the environment:

**Available Arguments:**

* ``--cpu``: Install CPU-only PyTorch (mutually exclusive with --cuda).
* ``--cuda``: Install CUDA-enabled PyTorch (default, and mutually exclusive with --cpu).
* ``--envpath PATH``: Path of the Python virtual environment to be installed. Default: ``./gflownet-env``
* ``--extras LIST``: Comma-separated list of extras to install. Default: ``all``. Options:
    
    - **dev**: dependencies for development, such as linting and testing packages.
    - **materials**: dependencies for materials applications, such as the Crystal-GFN.
    - **molecules**: dependencies for molecular modelling and generation, such the Conformer-GFN.
    - **all**: all of the above
    - **minimal**: none of the above, that is the minimal set of dependencies.

* ``--dry-run``: Print the summary of the configuration selected and exit.
* ``--help``: Show the help message and exit.

**Example Usage:**

For example, you may run:

.. code-block:: bash

    source install.sh --cpu --envpath ~/myenvs/gflownet-env --extras dev,materials

to install an environment on ``~/myenvs/gflownet-env``, with a CPU-only PyTorch and the dev and materials extras.

