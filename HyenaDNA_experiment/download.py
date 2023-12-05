import gdown
import urllib.request

print("Downloading 3utr.sequences.refseq_id.mirna_fc.pkl")
url = 'https://drive.google.com/file/d/13DS_s2ZXxBGz0DbWsCxRJWaoabUpFsr9/view?usp=drive_link'
output = '3utr.sequences.refseq_id.mirna_fc.pkl'
gdown.download(url, output, quiet=True, fuzzy=True)


print("Downloading standalone_hyenadna.py")
url = "https://raw.githubusercontent.com/HazyResearch/hyena-dna/main/standalone_hyenadna.py"
output = "standalone_hyenadna.py"
urllib.request.urlretrieve(url, output)