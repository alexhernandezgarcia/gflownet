Installation
============

This section provides detailed instructions on how to install **gflownet**. Ensure your system meets the following requirements before proceeding with the installation.

Requirements
------------

* **Python Version:** Python 3.10 is required.
* **CUDA Version:** CUDA 11.8 is needed for GPU acceleration.
* **Operating System:** The setup is primarily supported on Ubuntu. The installation may work on OSX with manual handling of package dependencies.

Quick Installation
------------------

For a quick installation, clone the repository and execute the setup script:

.. code-block:: bash

    git clone git@github.com:alexhernandezgarcia/gflownet.git
    cd gflownet
    ./setup_all.sh

Detailed Installation
---------------------

If you prefer a more controlled installation or need to customize the environment, follow these steps:

1. **Create a Virtual Environment:**
   
   .. code-block:: bash
   
       python3.10 -m venv ~/envs/gflownet  # Initialize your virtual environment.

2. **Activate the Virtual Environment:**

   .. code-block:: bash

       source ~/envs/gflownet/bin/activate  # Activate your environment.

3. **Install Ubuntu Prerequisites:**

   .. code-block:: bash

       ./prereq_ubuntu.sh  # Installs some packages required by dependencies.

4. **Install Python Prerequisites:**

   .. code-block:: bash

       ./prereq_python.sh  # Installs Python packages with specific wheels.

5. **Optional: Install Geometric Dependencies:**

   .. code-block:: bash

       ./prereq_geometric.sh  # OPTIONAL - for the molecule environment.

6. **Install gflownet:**

   .. code-block:: bash

       pip install .[all]  # Install the remaining elements of this package.

Optional Packages
-----------------

You can also install additional optional packages tailored for specific functionalities:

* **Development Tools:** Install with `dev` tag.
* **Materials Science Dependencies:** Install with `materials` tag.
* **Molecular Environment Packages:** Install with `molecules` tag.

The `all` tag installs all dependencies, which is the simplest and recommended option if you require full functionality.

