# Image Classification | MLP from Scratch, CNNs & Transfer Learning

An applied machine learning project that builds a multilayer perceptron (MLP) from scratch in NumPy, trains it on FashionMNIST, and runs a systematic set of experiments on depth, activations, regularization, momentum, normalization, and data augmentation. The best MLP is then compared against a PyTorch CNN and a frozen, ImageNet-pretrained ResNet-18, which is also tested on CIFAR-10 to study when transfer learning helps.

**Course:** COMP 551 – Applied Machine Learning
**Institution:** McGill University
**Project:** Mini Project 3 – Classification of Image Data with MLPs and CNNs
**Authors:** Ethan Gat, Anto Catanzaro, Marc Saliba

---

## Overview

[FashionMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) is a drop-in replacement for MNIST: 70,000 grayscale 28×28 images across 10 clothing classes. The data is split 54,000 / 6,000 / 10,000 (train / validation / test), with roughly 5,400 training images per class.

The core MLP, backpropagation, and minibatch optimizer were implemented from scratch with NumPy. The project asks four questions:

1. How do depth, activation function, regularization, momentum, normalization, and augmentation change MLP accuracy and overfitting?
2. How much does a CNN, which keeps the 2D structure of the image, improve on the best MLP?
3. Can a frozen ImageNet-pretrained ResNet-18 with a retrained fully connected head compete with a CNN trained from scratch?
4. Does pre-training help more on a dataset closer to ImageNet (CIFAR-10) than on grayscale clothing?

## Results

| Model | Val (%) | Test (%) | Train time (s) | Hardware |
| ----- | ------- | -------- | -------------- | -------- |
| Best MLP used in comparison (augmented) | 89.33 | 88.08 | 432.83 | CPU |
| MLP, momentum 0.9 (Experiment 4) | 89.62 | 88.80 | n/a | CPU |
| CNN (from scratch) | 91.98 | 91.33 | 141.59 | GPU |
| CNN with augmentation | 92.68 | 91.66 | 607.07 | GPU |
| Frozen ResNet-18 + 3-layer FC head (augmented) | 89.53 | 89.15 | 780.48 | GPU |

Headline findings:

- **One hidden layer is the big win.** Going from 0 to 1 hidden layer lifts test accuracy from 84.10% to 87.79%. A second layer adds no test accuracy and increases overfitting.
- **Dropout gives the best regularization trade-off.** It cuts the train/validation gap to 1.34% with almost no test-accuracy cost.
- **Normalization matters for stability.** Without it, training is erratic and test accuracy drops by 4.18 points.
- **The CNN beats the best MLP by about 3 points** (91.33% vs 88.08%) and trains in 142 s, but data augmentation adds only 0.3 points at over 4× the cost per epoch.
- **Frozen pre-trained features help where the data resembles ImageNet.** On CIFAR-10 the pre-trained ResNet-18 beats a scratch CNN by 7.7 points; on FashionMNIST it trails the scratch CNN by 2.5 points.

## Methodology

### Implementation

**MLP (NumPy)**
- Configurable number of hidden layers and units.
- Interchangeable activations: ReLU, Leaky ReLU (α = 0.01), tanh.
- Softmax output trained with cross-entropy.
- Kaiming/He initialization for ReLU-like activations, Xavier/Glorot for the rest.
- Backpropagation through every layer, with L1/L2 penalty gradients added to the weights.
- Dropout applied to hidden units during training.

**Optimizer (separate class)**
- Minibatch gradient descent with configurable learning rate, batch size, and epochs.
- Momentum, using a per-parameter velocity that resets between training runs.
- Gradient clipping to prevent exploding gradients.

**Data pipeline**
- A custom vectoriser flattens PIL images and optionally normalizes or augments them, returning NumPy arrays.
- The PyTorch CNN and ResNet use PyTorch's own tensor transforms instead.

**Default optimizer settings** (unless an experiment says otherwise): batch size 128, learning rate 0.01, 50 epochs, no clipping, no momentum.

### Experiments

Unless stated otherwise, MLP experiments use two hidden layers of 256 units with ReLU. Accuracies are in %.

#### 1. MLP depth

| Hidden layers | Train | Val | Test |
| ------------- | ----- | --- | ---- |
| 0 | 86.51 | 84.93 | 84.10 |
| 1 | 92.01 | 88.52 | 87.79 |
| 2 | 93.62 | 88.28 | 87.75 |

An MLP with zero hidden layers is mathematically a softmax regression, which is too simple for this data. Adding nonlinearity gives a large jump. The second layer only widens the train/validation gap (1.58 → 3.49 → 5.34 points).

#### 2. Activation functions

| Activation | Train | Val | Test |
| ---------- | ----- | --- | ---- |
| ReLU | 93.83 | 88.43 | 87.93 |
| tanh | 92.84 | 88.93 | 87.94 |
| Leaky ReLU | 93.43 | 88.10 | 87.65 |

All three land within about 0.3 points on test accuracy. tanh overfits least, and Leaky ReLU's validation accuracy plateaus earliest.

#### 3. Regularization

L1 with λ = 5×10⁻⁴, L2 with λ = 5×10⁻³, dropout with p = 0.3.

| Method | Train | Val | Test |
| ------ | ----- | --- | ---- |
| None | 93.62 | 88.47 | 87.53 |
| L1 | 86.67 | 85.92 | 84.64 |
| L2 | 90.29 | 88.05 | 87.19 |
| Dropout | 89.87 | 88.53 | 87.45 |

Every method narrows the train/validation gap. L1 narrows it most but underfits and costs about 3 points of test accuracy. Dropout keeps the highest validation accuracy, preserves test accuracy, and gives the most stable training curves.

#### 4. Momentum

| Momentum | Train | Val | Test |
| -------- | ----- | --- | ---- |
| 0.0 | 93.62 | 88.28 | 87.75 |
| 0.5 | 96.33 | 88.77 | 88.28 |
| 0.9 | 99.12 | 89.62 | 88.80 |

Higher momentum consistently improves test accuracy and speeds up convergence, but it also drives training accuracy toward 99% and increases overfitting.

#### 5. Input normalization

Learning rate 10⁻⁴, momentum 0.9, gradient clipping threshold 10.

| Inputs | Train | Val | Test |
| ------ | ----- | --- | ---- |
| Normalized | 87.15 | 86.47 | 85.22 |
| Unnormalized | 85.76 | 81.22 | 81.04 |

Unnormalized training is visibly unstable, with oscillating validation accuracy, and ends 4.18 test points lower.

#### 6. Data augmentation (MLP)

Horizontal flip plus color jitter, applied to training images only.

| Training data | Train | Val | Test |
| ------------- | ----- | --- | ---- |
| No augmentation | 95.04 | 88.82 | 88.14 |
| Augmentation | 92.16 | 89.33 | 88.08 |

Augmentation acts as a regularizer, shrinking the train/validation gap without changing test accuracy. It must preserve labels: vertical flips on MNIST would turn 6s into 9s.

#### 7. Convolutional neural network

- Two 3×3 conv layers (32 and 64 channels), each followed by ReLU and 2×2 max pooling.
- A 256-unit fully connected hidden layer with ReLU, then a 10-unit output layer.
- SGD with momentum 0.9, learning rate 0.05 (chosen from a validation sweep), batch size 128, 50 epochs.

| Model | Val | Test | Time (s) | Per epoch (s) |
| ----- | --- | ---- | -------- | ------------- |
| CNN | 91.98 | 91.33 | 141.59 | 2.8 |
| CNN + augmentation | 92.68 | 91.66 | 607.07 | 12.1 |

The CNN gains about 3 points over the best MLP because shared filters and pooling exploit the 2D structure that flattening throws away. It still overfits heavily (99.36% training accuracy), and validation peaks around epoch 7. Augmentation gives roughly +0.3 test and +0.7 validation points, moves the best epoch from 7 to 22, and makes each epoch more than 4× slower, which is not worth it on centered grayscale clothing.

The hyperparameter sweeps (8 epochs each, validation accuracy) showed:

| Sweep | Value | Val acc |
| ----- | ----- | ------- |
| Learning rate | 0.005 / 0.01 / **0.05** / 0.1 | 0.9103 / 0.9165 / **0.9198** / 0.9080 |
| Channels, kernel | 16-32 k3 / 32-64 k3 / 64-128 k3 | 0.9173 / **0.9198** / 0.9188 |

#### 8. Pre-trained model (transfer learning)

- ResNet-18 pre-trained on ImageNet, final FC layer removed, all convolutional layers frozen.
- Images resized to 64×64, replicated to 3 channels, normalized with ImageNet statistics.
- FC heads with 0 to 3 hidden layers were compared on cached features for 20 epochs; the 3-layer head (width 256) had the best validation accuracy. The 0.4-point gain from 2 to 3 layers is within single-run noise, so 2 layers would be nearly as good.
- Final model trained for 50 epochs with the same augmentation as Experiment 6.

| Model | Val | Test | Time (s) | Per epoch (s) |
| ----- | --- | ---- | -------- | ------------- |
| Frozen ResNet-18 + FC | 89.53 | 89.15 | 780.48 | 15.6 |

It beats the best MLP by about 1.1 points but trails the scratch CNN by 2.5 points: frozen ImageNet features cannot adapt to low-resolution grayscale clothing, while the CNN learns task-specific filters end to end. It is also the slowest model, because every augmented image must pass through the full frozen network each epoch.

#### 9. Second dataset: CIFAR-10

CIFAR-10 (45,000 / 5,000 / 10,000 train / validation / test) with the same frozen ResNet-18 + 3-layer head, the same augmentation, and a scratch CNN as reference (3 input channels).

| Model | Val | Test | Time (s) |
| ----- | --- | ---- | -------- |
| FashionMNIST, pre-trained | 89.53 | 89.15 | 780.48 |
| FashionMNIST, CNN from scratch | 92.68 | 91.66 | 607.07 |
| CIFAR-10, pre-trained | 78.40 | 78.39 | 718.70 |
| CIFAR-10, CNN from scratch | 71.48 | 70.68 | 629.46 |

Validation and test accuracy are taken at the best-validation epoch. Pre-training gives +7.7 points over scratch on CIFAR-10 and −2.5 points on FashionMNIST. CIFAR-10 contains natural colour objects that overlap with ImageNet concepts, while centered, grayscale, low-resolution clothing does not. The scratch CNN also overfits heavily on CIFAR-10.

#### 10. Error analysis (confusion matrix)

A 10×10 confusion matrix on the FashionMNIST test set (89.15% overall accuracy) shows where the errors concentrate:

- Easy classes: Trouser (976/1000), Bag (970), Ankle boot (965), Sandal (961), Sneaker (955).
- Hard class: Shirt, with only 680/1000 correct, mostly confused with T-shirt/top (143), Pullover (84), and Coat (56).
- T-shirt/top is most often mistaken for Shirt (100), and Coat for Pullover (92).

Errors cluster among upper-body garments with similar silhouettes, which motivates models that capture local spatial structure.

## Key Takeaways

| Question | Answer |
| -------- | ------ |
| Does depth help an MLP? | One hidden layer, yes; a second only adds overfitting. |
| Best regularizer? | Dropout: small gap, test accuracy preserved, stable training. |
| Is normalization optional? | No. Skipping it costs 4.18 test points and destabilizes training. |
| MLP or CNN? | CNN, about +3 points and faster on GPU. |
| Is augmentation worth it for the CNN? | Not here: +0.3 points for over 4× per-epoch cost. |
| When does pre-training pay off? | When the target data resembles the source data (CIFAR-10 yes, FashionMNIST no). |

## Technical Stack

- **Python**: implementation and experimentation
- **NumPy**: from-scratch MLP, backpropagation, and optimizers
- **PyTorch / torchvision**: CNN, pre-trained ResNet-18, CIFAR-10 and FashionMNIST loading
- **scikit-learn**: confusion matrix
- **Matplotlib**: training curves and comparison plots
- **Jupyter / Google Colab**: experimental workflow

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── image_classification.ipynb
├── src/
│   ├── cnn.py
│   ├── config.py
│   ├── mlp.py
│   ├── optimizer.py
│   ├── plotting.py
|   ├── preprocessing.py
|   ├── pretrained.py
|   ├── training.py
│   └── tuning.py
└── docs/
    ├── report.pdf
    └── figures/
```

## Getting Started

### Requirements

- Python 3.10+
- pip
- A GPU is recommended for the CNN and ResNet experiments. The from-scratch MLP runs on CPU. Also supports Apple Silicon MPS.

### Installation

Clone the repository:

```
git clone https://github.com/Anonymoose725/ML-Multilayer-Perceptron.git
cd ML-Multilayer-Perceptron
```

Install dependencies:

```
pip install -r requirements.txt
```

Launch Jupyter:

```
jupyter lab
```

Open `notebooks/image_classification.ipynb` and run the cells sequentially, or open it in Google Colab. FashionMNIST and CIFAR-10 are downloaded automatically.

## Limitations and Future Work

- **Single runs.** Every result comes from one run with one seed. The same MLP configuration (two hidden layers, 256 ReLU units, no regularization) gives 87.75%, 87.93%, and 87.53% test accuracy across experiments, so differences of a few tenths of a point, such as between activation functions, are within noise.
- **Mixed hardware.** MLPs ran on CPU and the CNN and ResNet on GPU, so training times are only indicative.
- **Different optimizer settings.** The MLP and CNN families use different momentum settings.
- **Best-epoch reporting.** CNN numbers are reported at the best epoch, while the MLP is reported at its final epoch. All models start overfitting early (CNN validation peaks around epoch 7, augmented and pre-trained models between epochs 22 and 36), so early stopping would save most of the compute.
- **Frozen backbone.** The ResNet-18 is fully frozen, so its FashionMNIST gap may come from the frozen features rather than from pre-training itself.

Potential extensions: repeat key experiments across multiple seeds, test depth jointly with dropout, use stronger augmentations such as random cropping, and fine-tune the last ResNet block.

## References

1. Zalando Research. [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist). Kaggle dataset.
2. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(56), 1929–1958.
3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
4. Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
6. Russakovsky, O., et al. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3), 211–252.

---

*Developed collaboratively as part of COMP 551 at McGill University, November 2026.*
