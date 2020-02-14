# Extreme Classification via Adversarial Softmax Approximation

This repository contains the code for our paper
* [R. Bamler and S. Mandt, Extreme Classification via Adversarial Softmax Approximation, ICLR 2020](bamler-mandt-adversarial-neg-sampling-iclr2020.pdf).


## Dependencies

This code was tested with TensorFlow version 1.15 on python 3.6.
The code to fit the auxiliary model was tested with Julia version 1.1.


## Directory Overview

- Directory `preprocess-extreme-predicton`:
  - Contains code to reproduce the exact binary representation of the data sets used in the paper.
  - Contains a jupyter notebook `pca.ipynb` that was used to generate the low-dimensional feature vectors for the auxiliary model as described in the paper.
- Directory `aux_model`:
  - Contains both Julia code to fit the auxiliary model and python code to use the fitted model during training of the main model, as described in the paper.
- File `train.py`:
  - The main file to train the proposed model. See paper for hyperparameters.
- Directory `main_model`:
  - Contains internal utilities used by `train.py`.
    You shouldn't usually need to run any of the python scripts in this directory manually.

## License

The source code in this repository is released under the [MIT License](LICENSE).
If you use this software for a scientific publication, please consider citing the following paper:
[R. Bamler and S. Mandt, Extreme Classification via Adversarial Softmax Approximation, ICLR 2020](bamler-mandt-adversarial-neg-sampling-iclr2020.pdf).


## Authors

* [Robert Bamler](https://robamler.github.io)
* [Stephan Mandt](http://www.stephanmandt.com/)
