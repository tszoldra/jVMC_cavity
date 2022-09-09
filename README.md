# jVMC_cavity
# Extension of the jVMC codebase for lattice-cavity systems


## Installation
1. We recommend you create a new conda environment to work with `jVMC_cavity`:

        conda create -n jvmc_cavity python=3.9
        conda activate jvmc

2. ``pip``-install the package

        pip install jVMC_cavity

3. If `mpi4py` installation with `pip` fails, try to install it with `conda`:

        conda install mpi4py

4. Install `qutip` manually:

        conda install qutip

To use GPU with CUDA, simply execute

        pip install jVMC_cavity[cuda]

<!---
442  conda create --name jaxgpu
443  conda activate jaxgpu
444  conda install mpi4py
445  python
446  conda install qutip
447  pip install qbism
448  python
449  conda install python=3.9.7
450  python --version
451  conda install qutip
452  cd jvmc/SpinPhoton2/vmc_jax/ # modified setup.py to not install jax - we will install it by hand
453  pip install -e .
454  pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
455  python -c "import jax"
conda install -c conda-forge mpi4py openmpi
-->
     
