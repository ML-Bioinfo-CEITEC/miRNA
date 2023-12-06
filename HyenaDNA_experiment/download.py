import gdown
import urllib.request
import os 

print("Downloading 3utr.sequences.refseq_id.mirna_fc.pkl")
url = 'https://drive.google.com/file/d/13DS_s2ZXxBGz0DbWsCxRJWaoabUpFsr9/view?usp=drive_link'
output = '3utr.sequences.refseq_id.mirna_fc.pkl'
gdown.download(url, output, quiet=True, fuzzy=True)


print("Downloading standalone_hyenadna.py")
url = "https://raw.githubusercontent.com/HazyResearch/hyena-dna/main/standalone_hyenadna.py"
output = "standalone_hyenadna.py"
urllib.request.urlretrieve(url, output)

directory = "./checkpoints"
model = "hyenadna-small-32k-seqlen"
print("Creating folder for storing downloaded model.")
if not os.path.exists(directory + "/" + model):
    os.makedirs(directory + "/" + model)
    
print("Downloading HyenaDNA model")
url = "https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/resolve/main/weights.ckpt"
output = directory + "/" + model + "/weights.ckpt"
urllib.request.urlretrieve(url, output)

print("Downloading config for a HyenaDNA model")
url = "https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/raw/main/config.json"
output = directory + "/" + model + "/config.json"
urllib.request.urlretrieve(url, output)