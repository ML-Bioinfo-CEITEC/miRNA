ENV_NAME="seed"

conda create -n $ENV_NAME python=3.8 ipykernel ipywidgets nb_conda_kernels 
source activate $ENV_NAME

conda install matplotlib scikit-learn biopython pandas gdown