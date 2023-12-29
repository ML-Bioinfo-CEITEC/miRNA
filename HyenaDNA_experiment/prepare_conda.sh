ENV_NAME="hyena-dna"

conda create -n $ENV_NAME python=3.8 ipykernel ipywidgets nb_conda_kernels 
source activate $ENV_NAME
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

wget https://raw.githubusercontent.com/HazyResearch/hyena-dna/main/requirements.txt
pip install -r requirements.txt
rm requirements.txt

# TODO: install Flash Attention - https://github.com/HazyResearch/hyena-dna?tab=readme-ov-file#dependencies

pip install comet_ml # for logging training
#pip install optuna # for hyperparameter tuning
pip install lightning # Pytorch Lightning for handling models