# VectorAdam Modified 

This is the repository for the modified VectorAdam algorithm.

## Requirements

To use VectorAdam implementation, you just need to have PyTorch installed in your environment.

The demo script are tested with PyTorch=1.11 and matplotlib=3.5.1. We also provide the environment file `vectoradam.yml`, which can be used to create a conda environment as in 
```
conda env create -f vectoradam.yml -n [env-name]
```
Note that this is tested on Ubuntu 18.04 only.

## Usage

To use VectorAdam in your project, 

```
optimizer = VectorAdam(
    [{'params': X, 'axis': -1}, 
     {'params': Y, 'axis':  1], 
     lr=lr, betas=betas, eps=eps))
```

The above example will apply VectorAdam's vector-wise operations to X along the last axis and Y along the 1st axis, with specified learning rate, betas and epsilon hyperparameters.