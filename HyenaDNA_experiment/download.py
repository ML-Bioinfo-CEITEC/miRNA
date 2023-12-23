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