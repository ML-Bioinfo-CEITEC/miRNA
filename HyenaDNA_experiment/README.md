## Prerequisities

- unix operating system
- python (idealy 3.8)
- conda

## Environment 

We use `conda` environment to keep our packages in correct versions. There are a few steps needed to setup a proper `conda` environment and they are in the `prepare_conda.sh` script. In the first line of script, you can change the name of environment.

Run:

`. prepare_conda.sh`

After this step, you should have activated conda environment with a name `hyena-dna`.

## External files

`python download.py`


## Data preprocessing

`python src/data_preprocessing.py data/3utr.sequences.refseq_id.mirna_fc.pkl data`