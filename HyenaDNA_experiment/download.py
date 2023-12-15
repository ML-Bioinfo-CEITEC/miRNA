import gdown
import urllib.request
import os 

data_dir = "./data"
if not os.path.exists(data_dir + "/" + model):
    print("Creating folder for storing data.")
    os.makedirs(data_dir + "/" + model)

print("Downloading 3utr.sequences.refseq_id.mirna_fc.pkl")
url = 'https://drive.google.com/file/d/13DS_s2ZXxBGz0DbWsCxRJWaoabUpFsr9/view?usp=drive_link'
output = 'data/3utr.sequences.refseq_id.mirna_fc.pkl'
gdown.download(url, output, quiet=True, fuzzy=True)

print("Downloading standalone_hyenadna.py")
url = "https://raw.githubusercontent.com/HazyResearch/hyena-dna/main/standalone_hyenadna.py"
output = "src/standalone_hyenadna.py"
urllib.request.urlretrieve(url, output)

downloaded_model_dir = "./checkpoints"
model = "hyenadna-small-32k-seqlen"
if not os.path.exists(downloaded_model_dir + "/" + model):
    print("Creating folder for storing downloaded model.")
    os.makedirs(downloaded_model_dir + "/" + model)
    
print("Downloading HyenaDNA model")
url = "https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/resolve/main/weights.ckpt"
output = downloaded_model_dir + "/" + model + "/weights.ckpt"
urllib.request.urlretrieve(url, output)

print("Downloading config for a HyenaDNA model")
url = "https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/raw/main/config.json"
output = downloaded_model_dir + "/" + model + "/config.json"
urllib.request.urlretrieve(url, output)