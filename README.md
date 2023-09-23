# Summary

This repository contains the code to reproduce the results of MRPA-LegNet, a variant of LegNet [doi:10.1093/bioinformatics/btad457] specifically modified and optimized for predicting gene expression from human massive parallel reporter assays in K562, HepG2, and WTC11 cell lines.

# Model training and cross-validation
Please install all requirements with `environment.txt` or `environment.yml` with `conda` (documentation[https://conda.io/projects/conda/en/latest/user-guide/install/index.html]) `mamba` (documentation)[https://mamba.readthedocs.io/en/latest/mamba-installation.html]:
```
cd human_legnet
mamba env create -f envs/environment.yml 
```

# Model training and cross-validation

10-fold cross-validation was used for the model training and testing. The respective script is python vikram/human_legnet/core.py which be used as follows:

```
python core.py --model_dir <target dir to save the models checkpoint files> --data_path <experiment data tables used in study> --epoch_num <epochnum> --use_shift --reverse_augment
```

Please use --help for more details regarding.

This script was also used to test the impact of data augmentation on the model performance, e.g, adding --use_reverse_channel will add the respective channel to the input. In the study, the shift was set to the size of the forward adapter (21 bp).

# Assessing LegNet performance in predicting the allele-specific events

To get predictions of all cross-validation models we used asb_predict.py for ADASTRA datasets (allele-specific transcription factor binding, https://adastra.autosome.org) and coverage_predict.py for UDACHA (allele-specific chromatin accessibility, https://udacha.autosome.org) datasets.

Command-line format:
```
python asb_predict.py --config <model config> --model <models dir> --asb_path <path to the ADASTRA dataset> --genome <path to human genome> --out_path <out file> --device 0 --max_shift 0
```
```
python coverage_predict.py --config <model config> --model <models dir> --cov_path <path to the UDACHA dataset> --genome <path to human genome>  --out_path <out file>  --device 0 --max_shift 0
```
The datasets are available in the repository, see the datasets/asb and datasets/coverage folders.

The scripts compare the predicted and the real effect direction of single-nucleotide variants for the alternative against the reference allele at allele-specific regulatory sites. That is whether the reference or the alternative allele is preferable for transcription factor binding or for chromatin accessibility. 

The post-processing of predictions is performed with Jupyter-Notebook variant_annotation.ipynb.

Estimating the performance metrics (Fisher's exact test, R-squared calculation) for the processed predictions is performed using R script analyze_concordance.R

# Performing LegNet predictions for a user-supplied fasta

It is possible to run a LegNet model against a user-supplied fasta to obtain predictions for each sequence in it. This can be achieved with fasta_predict.py.

Example command:
```
python fasta_predict.py --config <model config> --model <model checkpoint> --fasta <path to fasta> --out_path <out path> --device 0 
```

Please note, that this is a fairly basic wrapper, and this version of LegNet is optimized to sequence length of 230bp.
LegNet technically will work for different sequence lengths as it uses global average pooling. 
However, if a sequence size  differs from 230 significantly, the resulting performance will be likely rather low. 
Also, due to the per-batch prediction, it is impossible to predict scores for sequences of different sizes if the batch size is not equal to 1.

