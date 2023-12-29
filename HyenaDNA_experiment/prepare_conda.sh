ENV_NAME="hyena-dna"

conda create -n $ENV_NAME python=3.8 ipykernel ipywidgets nb_conda_kernels 
source activate $ENV_NAME
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install matplotlib
pip install scikit-learn
pip install biopython
pip install pandas
pip install gdown # for downloading initial data
pip install comet_ml # for logging training
pip install lightning # Pytorch Lightning for handling models
pip install transformers # for using HyenaDNA model from HF