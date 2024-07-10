# Data sources

## data/target_scan_8 (downloaded on 1.6.2024)
[https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi](https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi)

### Predicted Targets context++ scores (default predictions) - (17.67 MB)
Context++ scores for predicted (conserved) targets of conserved miRNA families	Gene ID, Gene Symbol, Transcript ID, Species ID, miRNA, Site type, UTR start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, predicted relative KD -- updated 23 May 2022	1,397,979


___________________________________________________________________________________________________________________
[https://www.targetscan.org/vert_80/vert_80_data_download/Nonconserved_Site_Context_Scores.txt.zip](https://www.targetscan.org/vert_80/vert_80_data_download/Nonconserved_Site_Context_Scores.txt.zip)

### Nonconserved site context++ scores - (542.49 MB)	
Context++ scores, KDs for all nonconserved miRNA sites	Gene ID, Gene Symbol, Transcript ID, Species ID, miRNA, Site type, UTR start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, predicted relative KD -- updated 23 May 2022	38,497,660


___________________________________________________________________________________________________________________
[https://www.targetscan.org/vert_80/vert_80_data_download/Conserved_Site_Context_Scores.txt.zip](https://www.targetscan.org/vert_80/vert_80_data_download/Conserved_Site_Context_Scores.txt.zip)

### Conserved site context++ scores - (18.6 MB)	
Context++ scores, KDs for all conserved miRNA sites	Gene ID, Gene Symbol, Transcript ID, Species ID, miRNA, Site type, UTR start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, predicted relative KD -- updated 23 May 2022	1,468,778


___________________________________________________________________________________________________________________
[https://www.targetscan.org/vert_80/vert_80_data_download/Gene_info.txt.zip](https://www.targetscan.org/vert_80/vert_80_data_download/Gene_info.txt.zip)

### Gene info - (0.62 MB)	
Information about human genes	Transcript ID, Gene ID, Gene symbol, Gene description, Species ID, Number of 3P-seq tags + 5, Representative transcript?	28,353


## data/fold_change/mirna_fcs.csv (downloaded on 1.6.2024)
[https://doi.org/10.7554/eLife.05005.025](https://doi.org/10.7554/eLife.05005.025)


### Supplementary file 2
Normalized values for fold changes (log2) of mRNAs detectable in the seven datasets examining the response of transfecting miRNAs into HCT116 cells.


## data/GRCh37.p13 hg19
### utr_sequences_hg19.txt
From ensembl biomart using the following command or selecting these attributes in the UI browser of GRCh37 - human genome 19. 

```wget -O utr_sequences_hg19.txt 'http://grch37.ensembl.org/biomart/martservice?query=<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query  virtualSchemaName = "default" formatter = "FASTA" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" ><Dataset name = "hsapiens_gene_ensembl" interface = "default" ><Attribute name = "ensembl_gene_id" /><Attribute name = "ensembl_gene_id_version" /><Attribute name = "3_utr_start" /><Attribute name = "3_utr_end" /><Attribute name = "3utr" /><Attribute name = "chromosome_name" /><Attribute name = "external_gene_name" /><Attribute name = "start_position" /><Attribute name = "end_position" /><Attribute name = "strand" /><Attribute name = "transcript_start" /><Attribute name = "transcript_end" /><Attribute name = "transcript_length" /><Attribute name = "ensembl_transcript_id" /><Attribute name = "ensembl_transcript_id_version" /></Dataset></Query>'```