import os
import argparse
import zipfile
import gdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                         help='Path to directory to store data to.')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("Downloading data processed after we found bug in flipping sequences on negative strand")
    url = 'https://drive.google.com/file/d/1j8iFK1T0Inm1ealmVvFgzN7lYx1KXjgu/view?usp=drive_link'
    output = args.data_dir + '/miRNA_target_scanning.zip'
    gdown.download(url, output, quiet=True, fuzzy=True)
    print("Unzipping")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(args.data_dir)
    print("Removing zip archive")
    os.remove(output)
    
if __name__ == '__main__':
    main()