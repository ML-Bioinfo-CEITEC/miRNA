# Data sources

### Predicted Targets context++ scores (default predictions)
Source: [https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi](https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi)\
Destination: data/target_scan_8\
Downloaded on: 1.6.2024\
Source update on: 23.5.2022\
Size: 17.67 MB\
Description: Context++ scores for predicted (conserved) targets of conserved miRNA families\
Columns: Gene ID, Gene Symbol, Transcript ID, Species ID, miRNA, Site type, UTR start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, predicted relative KD\
Number of rows: 1,397,979


___________________________________________________________________________________________________________________

### Conserved site context++ scores (All predictions for representative transcripts)
Source: [https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi](https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi)\
Destination: data/target_scan_8\
Downloaded on: 1.6.2024\
Source update on: 23.5.2022\
Size: 18.6 MB\
Description: Context++ scores, KDs for all conserved miRNA sites	Gene ID, Gene Symbol, Transcript ID, Species ID, miRNA, Site type, UTR start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, predicted relative KD\
Number of rows: 1,468,778

___________________________________________________________________________________________________________________

### Nonconserved site context++ scores (All predictions for representative transcripts)
Source: [https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi](https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi)\
Destination: data/target_scan_8\
Downloaded on: 1.6.2024\
Source update on: 23.5.2022\
Size: 542.49 MB\
Description: Context++ scores, KDs for all nonconserved miRNA sites	Gene ID, Gene Symbol, Transcript ID, Species ID, miRNA, Site type, UTR start, UTR end, context++ score, context++ score percentile, weighted context++ score, weighted context++ score percentile, predicted relative KD\
Number of rows: 38,497,660


___________________________________________________________________________________________________________________

### Gene info (Gene and miRNA annotations)	
Source: [https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi](https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi)\
Destination: data/target_scan_8\
Downloaded on: 1.6.2024\
Source update on: 23.5.2022\
Size: 0.62 MB\
Description: Information about human genes	Transcript ID, Gene ID, Gene symbol, Gene description, Species ID, Number of 3P-seq tags + 5, Representative transcript?\
Number of rows: 28,353

___________________________________________________________________________________________________________________

### mirna_fcs (Supplementary file 2)
Source: [https://doi.org/10.7554/eLife.05005.025](https://doi.org/10.7554/eLife.05005.025)\
Destination: data/fold_change\
Downloaded on: 1.6.2024\
Source update on: 23.5.2022\
Size: 504 KB\
Description: Normalized values for fold changes (log2) of mRNAs detectable in the seven datasets examining the response of transfecting miRNAs into HCT116 cells. Used by TargetScan 7 and 8 as a test set. miRNAs: hsa-miR-16-5p,	hsa-miR-106b-5p,	hsa-miR-200a-3p,	hsa-miR-200b-3p,	hsa-miR-215-5p,	hsa-let-7c-5p,	hsa-miR-103a-3p\
Number of rows: 8,373

___________________________________________________________________________________________________________________

### utr_sequences_hg19.txt
Source: From ensembl biomart using the following command or selecting these attributes in the UI browser of GRCh37 - human genome 19. 
```wget -O utr_sequences_hg19.txt 'http://grch37.ensembl.org/biomart/martservice?query=<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query  virtualSchemaName = "default" formatter = "FASTA" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" ><Dataset name = "hsapiens_gene_ensembl" interface = "default" ><Attribute name = "ensembl_gene_id" /><Attribute name = "ensembl_gene_id_version" /><Attribute name = "3_utr_start" /><Attribute name = "3_utr_end" /><Attribute name = "3utr" /><Attribute name = "chromosome_name" /><Attribute name = "external_gene_name" /><Attribute name = "start_position" /><Attribute name = "end_position" /><Attribute name = "strand" /><Attribute name = "transcript_start" /><Attribute name = "transcript_end" /><Attribute name = "transcript_length" /><Attribute name = "ensembl_transcript_id" /><Attribute name = "ensembl_transcript_id_version" /></Dataset></Query>'```\
Destination: data/"GRCh37.p13 hg19"\
Downloaded on: 1.6.2024\
Source update on: \
Size: 101.9 MB\
Description: The 3'UTR sequences from the human genome version 19. The file also contains the following attributes: ensembl_gene_id, ensembl_gene_id_version, 3_utr_start, 3_utr_end, 3utr, chromosome_name, external_gene_name, start_position, end_position, strand, transcript_start, transcript_end, transcript_length, ensembl_transcript_id, ensembl_transcript_id_version\
Number of rows: 1,611,440
