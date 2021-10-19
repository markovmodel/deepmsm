# deepmsm

UPDATE:

New example code was added to build your own deep reversible hierarchical MSM with attention mechanism:
"Attention_and_coarse_graining.ipynb".
and your deep reversible MSM with additional experimental values:
"Observable_training.ipynb"
The code there needs additionally the deeptime package and represents a pre-alpha version how the code will
be integrated into deeptime (see https://deeptime-ml.github.io/latest/index.html). It also includes the functionality to
monitor the performance of the model during the training via tensorboard. If you wanna use the feature you need to install
it too.


Code to reproduce results for the paper: Progress in deep Markov State Modeling: Coarse graining and experimental data restraints

The code is presented in two notebooks and a helper file network.py and is applied for the Villin dataset. 

The file "Villin_attention_and_coarse_graining.ipynb" covers the parts:
1. Comparing the results of a revDMSM against a VAMPnet
2. Plotting network graphs of the resulting stationary distribution and transition matrix with representative structures
3. Estimating implied timescales and performing the CK-test
4. Estimating the eigenfunctions and plot them with representative structures
5. Training the coarse-graining layers
6. Plotting the hierarchical model

The file "Villin_training_with_experimental_observables.ipynb" covers:
1. In the absence of real experimental values, estimate "true" values of the original Villin trajectory
2. Manipulate the data by removing a percentage of the folding and unfolding events from the data
3. Train a VAMPnet as a initial network
4. Train a revDMSM without further information of the true values
5. Train a revDMSM with further knowledge of the true values, where it can be switched between three classes of observables

In order to run the code the following package are required:
- Python 3
- Jupyter Notebook or Jupyter Lab
- NumPy
- Matplotlib
- PyTorch
For saving representative structures 
- MDTraj
For initializing the coarse-graining matrices
-PyEMMA

The authors aim to update the code to be compatible with the deeptime package (see https://deeptime-ml.github.io/latest/index.html)
