# BHIT
Bayesian High-order Interaction Toolkit (BHIT) uses a novel Bayesian computational method with a Markov Chain Monte Carlo (MCMC) search for detecting epistatic interactions among single nucleotide polymorphisms (SNPs).

You can read the full paper and have a better understanding about what we have done by [clicking here](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-015-2217-6), which was published several years ago and free for public to view.

We now support both **Numpy** and **Tensorflow** version, yet we have not fully tested them. If you want to see how it works, an interactive version with Jupyter notebook is available [here](notebook/).

## Introduction

Epistasis (gene-gene interaction), including high-order interaction among more than two genes, often plays important roles in complex traits and diseases, but current GWAS (Genome Wide Association Study) analysis usually just focuses on additive effects of Single Nucleotide Polymorphisms (SNPs).  The lack of effective computational modelling of high-order functional interactions often leads to significant under-utilization of GWAS data. 

BHIT first builds a Bayesian model on both continuous data and discrete data, which is capable of detecting high-order interactions in SNPs related to case—control or quantitative phenotypes. 

## Key Strengths

-  With the advanced Bayesian model using MCMC search, BHIT can efficiently explore high-order interactions;
-  BHIT can handle both continuous and discrete phenotypes, and the interaction within or between phenotypes and genetic data can also be detected. 

## Method and Algorithm

As we mentioned before, the complete paper is [here](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-015-2217-6). Alternatively, you can just understand the skeleton by reading [this notebook](https://nbviewer.jupyter.org/urls/gitee.com/candydog/BHIT/raw/master/notebook/np_bhit.ipynb).

## Acknowledgement

I worked on improving previous BHIT program when I came to the University of Missouri, where I had a good time with Prof. Dong Xu and Dr. Juexin Wang. 

**All rights reserved by Digital Biology Lab (DBL) at University of Missouri.**