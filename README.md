## EXPLORA: Efficient Exemplar Subset Selection for Complex In-context Numerical Reasoning

This repository contains the implementation details of our Exemplar Selection Algorithm for In-context Learning.



## Requirements
The code is written for python `3.9`, but should work for other version with some modifications.
Create a conda environment with python version `3.9`.  Install cudatoolkit according to gpu compatibility.
```
pip install -r requirements.txt
```

## Data Preparation

Please download the embeddings for the datasets used from this [Link](https://drive.google.com/drive/folders/1pWFBRMBsnWS5Ty1owK2lyIy7vPCIO4_R?usp=sharing) 

## Python script overview

`AquaRat/explora+SC.py` - It contains the code for exemplar selection on AquaRat using Explora approach with self-consistency decoding.\
`FinQA/explora+SC.py` - It contains the code for exemplar selection on FinQA using Explora approach with self-consistency decoding.\
`GSM8K/explora+SC.py` - It contains the code for exemplar selection on GSM8K using Explora approach with self-consistency decoding.\
`TabMwp/explora+SC.py` - It contains the code for exemplar selection on TabMWP using Explora approach with self-consistency decoding.

