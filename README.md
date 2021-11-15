# **Categorical Perception: A Groundwork for Deep Learning**
### Full Python 3 source code used for the results presented in our [paper](https://doi.org/10.1162/neco_a_01454).

Reference :   
Bonnasse-Gahot, L., & Nadal, J- P. (2020). Categorical Perception: A Groundwork for Deep Learning.
Neural Computation 2021.   
doi: https://doi.org/10.1162/neco_a_01454   
arXiv preprint: https://arxiv.org/abs/2012.05549

Corresponding author: lbg@ehess.fr

#### Dependencies

See requirements.txt for the full list of packages used in this work. This file provides the exact version that was used, but the code is expected to work with other versions as well. The main dependency is tensorflow v2.4.1.

We use an NVIDIA GeForce GTX 1080, with drivers v465.19.01, with CUDA v11.3 and CUDNN v8.1.1. Results might slightly differ depending on the machine, the drivers, and the version of the packages that are used -- although we expect no significant difference in the results.

#### Installation
The following procedure should work on most recent computers.
```
conda create --name cpdl python=3.7
source activate cpdl
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=cpdl
```

Once the environment is installed, you can run each of the following notebooks by first running `jupyter notebook` or `jupyter lab`.

#### List of the Python notebooks and script

[`gaussian1d_mlp.ipynb`](gaussian1d_mlp.ipynb)   
Python notebook for the one dimensional toy example, Section 3.1.1, reproducing Figure 1 of the paper.

[`gaussian2d_mlp_dynamics.ipynb`](gaussian2d_mlp.ipynb)   
Python notebook for the two dimensional toy example, Section 3.1.2, reproducing Figure 2 of the paper.

[`mnist_train_autoencoder.ipynb`](mnist_train_autoencoder.ipynb)   
Python notebook for training the autoencoder on the MNIST training set, used in Section 3.2 for the Figures 3, 4 and 5 of the paper.

[`mnist_mlp_changes_in_neural_representation.ipynb`](mnist_mlp_changes_in_neural_representation.ipynb)
Python notebook for reproducing the results presented in Section 3.2.2, Figure 4 (Changes in the neural representation following learning of categories).

[`mnist_mlp_categorical_perception.ipynb`](mnist_mlp_categorical_perception.ipynb)   
Python notebook for reproducing the results presented in Section 3.2.3, Figures 3 (Examples of image continua) and 5 (Gradual categorical perception across layers), as well as Figure B.1.

[`mnist_mlp_categoricality.ipynb`](mnist_mlp_categoricality.ipynb)   
Python notebook for reproducing the results presented in Section 3.2.4, Figure 6 (Categoricality as a function of layer depth, using the MNIST dataset), panel a, as well as Figure D.2, panel a.

[`mnist_cnn_categoricality.ipynb`](mnist_cnn_categoricality.ipynb)   
Python notebook for reproducing the results presented in Section 3.2.4, Figure 6 (Categoricality as a function of layer depth, using the MNIST dataset), panel b, as well as Figure D.2, panel b.

[`cat_dog_train_cnn.ipynb`](cat_dog_train_cnn.ipynb)   
Python notebook for training a deep convolutional neural network to classify natural images of cats and dogs (using the Kaggle Dogs vs. Cats database), used in Figure 7 (Categorical perception of a cat/dog circular continuum), Section 3.3.1.

[`cat_dog_categorical_perception.ipynb`](cat_dog_categorical_perception.ipynb)   
Python notebook for reproducing Figure 7 (Categorical perception of a cat/dog circular continuum), presented in Section 3.3.1, and Figure C.1.

[`imagenet_vgg16_categoricality.ipynb`](imagenet_vgg16_categoricality.ipynb)   
Python notebook for reproducing the results presented in Section 3.3.2, Figure 8 (Categoricality as a function of layer depth, using the ImageNet dataset).

[`figure_estimation_posterior.ipynb`](figure_estimation_posterior.ipynb)   
Python notebook for reproducing Figure A.1 (Increasing Fisher information $F_\text{code}$ decreases the probability of error due to noise in the neural processing)

[`cifar10_mlp_dropout.py`](cifar10_mlp_dropout.py)   
Python script for reproducing the results presented in Figure D.1 (Classification accuracy on the CIFAR-10 image dataset (test set) using a multi-layer perceptron with varying levels of dropout.).
In our work we use ten trials running the script with random seeds ranging from 1 to 10. This can be achieved thanks to the following bash code:
```
for i in {1..10}
do
  python cifar10_mlp_dropout.py --seed $i;
done
```

[`cifar10_dropout_figure.ipynb`](cifar10_dropout_figure.ipynb)   
Python notebook for reproducing D.1 (Classification accuracy on the CIFAR-10 image dataset (test set) using a multi-layer perceptron with varying levels of dropout.).
