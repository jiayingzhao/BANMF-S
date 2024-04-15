# BANMF-S: a blockwise accelerated non-negative matrix factorization framework with structural network constraints for single cell imputation


## Description
This repository is an implementation for BANMF-S ( **B**lockwise **A**ccelerated **N**on-negative **M**atrix **F**actorization framework with **S**tructural network constraints), a scalable computational method for single cell imputation.
 

## Install
BANMF-S (version 1.0.2) requires Python (version >= 3.8.0) (https://www.python.org) and several dependancies:

- numpy (1.24.3)
- scipy (1.10.1)
- pandas (1.3.5)
- scikit-learn (1.2.2)
- networkx (3.1)

Users may also create a new conda environment named banmfs (or any other name) from the provided *requirements.txt* file:

```
conda create -n banmfs -f requirements.txt
```

or from the provided *environment.yml* file:

```
conda env create -f environment.yml
```


The construction of gene similarity network is implemented on R (https://www.r-project.org), with several packages required:

- STRINGdb
- igraph
- parallel

For a complete list of attached packages and their corresponding versions, please go to the text file *RsessionInfo.txt* under this repository.




Afterwards, clone this repository to your local machine by:

```
git clone https://github.com/jiayingzhao/BANMF-S.git /path/to/your/dir

cd /path/to/your/dir/BANMF-S
```

## Gene and Cell Similarity Networks

To implement BANMF-S, you need to save the "gene-by-cell" and "cell-by-gene" normalized log-transformed expression matrix under one folder, and name them into *genebycell_lognorm.csv* and *cellbygene_lognorm.csv*. Then follow the instructions in this vignette, the Laplacian matrix of gene similarity network (named by *gsim.csv*) and the adjacency matrix of cell similarity network (named by *csim.csv*) will be created under the same folder. See our provided *data* folder for example.



Users should obtain the Laplacian matrix of gene similarity network in R by running the following snipet:
```{r eval = FALSE}

setwd('/path/to/your/BANMF-S')
source('./src/function.R')
dat <- read.csv('./data/genebycell_lognorm.csv', row.names = 1)
ppi <- getPPI(expr = dat, species = 'human', genFlag='symbol')
genesimmat <- compJIppi(dat = ppi, numcores = 2)
write.csv(genesimmat, './data/gsim.csv',quote=F)

```

The adjacency matrix of cell similarity network can be obtained from shell by running:

```
cd /path/to/your/BANMF-S

# -i : the input folder containing cellbygene_lognorm.csv file, the csim.csv file will be created in the same folder.

python ./src/proc_minST.py -i /path/to/your/BANMF-S/data 

```


## Run BANMF-S

```
cd /path/to/your/BANMF-S/

# B: the number of the split of blocks, equals to the number of processes. 
python src/main.py -i /path/to/your/data/ \
   -o /path/to/your/data/ \
   --B 5
```

For parameters and default values, please run the following cammand in shell

```
cd /path/to/your/BANMF-S/

python main.py -h
```
The results can be found under the input folder:
- banmfs_Ximp.csv : imputed cell-by-gene expression matrix.
- banmf_W.csv: cell matrix.
- banmfs_H.csv: gene matrix.
- banmfs_log.txt and banmfs_err.csv: loss recordings

## Citation




## Contact

For any enquiries, please contact Jiaying Zhao (jyzhao@connect.hku.hk).
