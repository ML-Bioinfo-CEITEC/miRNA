import gdown
import urllib.request
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                     help='Path to directory to store data to.')
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(data_dir):
    print("Creating folder for storing data.")
    os.makedirs(data_dir)

print("Downloading 3utr.sequences.refseq_id.mirna_fc.pkl")
url = 'https://drive.google.com/file/d/13DS_s2ZXxBGz0DbWsCxRJWaoabUpFsr9/view?usp=drive_link'
output = data_dir + '/3utr.sequences.refseq_id.mirna_fc.pkl'
gdown.download(url, output, quiet=True, fuzzy=True)

print("Downloading data processed after we found bug in flipping sequences on negative strand")
url = 'https://drive.google.com/file/d/1j8iFK1T0Inm1ealmVvFgzN7lYx1KXjgu/view?usp=drive_link'
output = data_dir + '/miRNA_target_scanning.zip'
gdown.download(url, output, quiet=True, fuzzy=True)