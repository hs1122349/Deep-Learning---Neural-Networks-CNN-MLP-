# Deep Learning, EDC Paris

Coursework and end-of-course submissions for the Deep Learning class taught by **Pierre Dossantos-Uzarralde** at EDC Paris Business School (MSC2 / PGE5 DSBA).

This repository collects everything I worked on for the class: hand-derived backpropagation exercises, a multilayer perceptron built from scratch in NumPy, the same network reimplemented in PyTorch for comparison, and an architecture-search study on a noisy classification problem.

## Authors

**Hedi Souki**, in collaboration with **Stéphanie Abi Saab**.  
Group submissions for the EDC Paris Deep Learning course, 2025-2026.

## About the class

The Deep Learning course is taught over four days by Pierre Dossantos-Uzarralde and is structured around progressive end-of-course submissions (ECS):

| Day | Main topic | Submission |
|----:|------------|------------|
| 1 | Regression and gradient descent | ECS 1 and ECS 2 |
| 2 | Single-layer perceptron and MLP | ECS 3 |
| 3 | CNN and RNN | ECS 4 |
| 4 | Final project | Oral presentation / Report Document|

All deliverables are submitted as Jupyter notebooks (`.ipynb`) following the naming convention `studentname_ECS_nb_EDC_Gxx`.

## What's in this repo

```
.
├── README.md
├── ecs1_regression/
│   └── hedi-souki_ECS_01_EDC_Gxx.ipynb
├── ecs2_gradient_descent/
│   └── hedi-souki_ECS_02_EDC_Gxx.ipynb
├── ecs3_single_layer/
│   ├── neurones_deep_learning.py            # 1-hidden-layer MLP, activation comparison
│   └── hedi-souki_ECS_03_EDC_Gxx.ipynb
├── ecs4_mlp/
│   ├── Deep_Learning_TD4_Hedi_SOuki.ipynb   # main notebook, architecture search
│   ├── td4_pytorch_mlp.py                   # PyTorch comparison (SGD vs Adam)
│   └── TD4-MLP-Exo1.pdf                     # original exercise
├── td1_hand_calculations/
│   └── TD1.pdf                              # forward and backward pass by hand
└── final_project/
    ├── hedi-souki_FinalProject_EDC_Gxx.ipynb   # pneumonia detection from chest X-rays
    └── data/                                   # not committed, dataset downloaded locally from Kaggle
```

## The work, in order

### TD1, forward and backward propagation by hand

Two small exercises done with pen and paper before any code: a forward pass through a tiny two-input, one-hidden-layer MLP with ReLU activations, then a full backward pass on a one-neuron-per-layer network using the chain rule. The goal was to feel exactly what every term in the back-propagation formulas refers to before letting NumPy or PyTorch hide the work.

### ECS 3, single-layer perceptron in NumPy

A from-scratch implementation of a one-hidden-layer MLP applied to the `make_circles` dataset from scikit-learn. The script (`neurones_deep_learning.py`) iterates over different numbers of hidden neurons and four activation functions (linear, sigmoid, tanh, ReLU), then plots test accuracy across configurations.

The point of the exercise was less about beating a benchmark and more about seeing concretely how:
* linear activations cannot solve a non-linearly-separable problem, no matter how many neurons you stack;
* sigmoid and tanh suffer from vanishing gradients on deeper networks;
* ReLU is the modern default but does not always win on small, low-dimensional problems.

### ECS 4, the 4-layer MLP

The main piece of the course so far. We were asked to build a multilayer perceptron with two hidden layers (16 and 8 neurons, ReLU activations, sigmoid output) and train it to classify points from two concentric noisy rings. The work was broken into three parts:

1. **A flexible MLP implemented from scratch in NumPy.** Forward propagation, binary cross-entropy loss, back-propagation derived by hand, gradient descent, all written in a way that handles any number of hidden layers.

2. **A PyTorch reimplementation of the same network.** Same architecture, same data, same hyperparameters. Used to compare training speed and to study the impact of choosing SGD versus Adam as the optimizer. The PyTorch version replaces the entire hand-written backprop code with three lines (`zero_grad`, `loss.backward()`, `step`), which makes the value of autograd very obvious.

3. **An architecture search.** Starting from the baseline configuration suggested by the exercise, we varied one hyperparameter at a time:
   * number of hidden neurons,
   * number of hidden layers,
   * number of training epochs.

   On the original dataset (radii 1 and 3, noise 0.1), every configuration reaches 100% test accuracy, which makes the comparison meaningless. We therefore generated a harder version of the same problem (closer rings, more noise, fewer points) and re-ran the trials. The most interesting finding is that the naive "best of each independent trial" combination performs poorly, because hyperparameters interact. A small joint grid search over the top candidates from each trial found a configuration that meaningfully beats the baseline.

The notebook is structured as a narrative: each trial has a motivation, the experiment, a results table or chart, and a short reading of what the numbers say.

### Final project, pneumonia detection from chest X-rays

The course closes with a free project to be presented orally in groups of two. We chose the **Chest X-Ray Images (Pneumonia)** dataset published by Paul Mooney on Kaggle, around 5,800 pediatric chest X-rays labelled as Normal or Pneumonia, already split into train, validation and test folders. It is a real medical imaging dataset with known issues (very small validation set, class imbalance, sensitive symmetry properties), which makes it a good playground for the kind of architectural and training-tricks comparison we want to run.

The project is built around two reference models and a controlled ablation study:

1. **A convolutional network built from scratch in PyTorch.** Three to four conv blocks, batch normalization, dropout, binary cross-entropy loss. This is our baseline and the reference we want to beat.

2. **Transfer learning from a pretrained ResNet18.** The backbone is initialized with ImageNet weights, the final classifier head is replaced for the binary task, and we fine-tune. This is the modern default for medical imaging on small datasets, and the gap with the from-scratch model is exactly the story we want to tell.

3. **An ablation sweep on top of the two baselines.** Same methodology as the TD4 architecture search: vary one thing at a time and measure how much it actually contributes. The dimensions we explore are:
   * input image size (128 vs 224),
   * data augmentation strategy (none, horizontal flips and rotations only, heavier augmentation), being careful to avoid vertical flips since left-right asymmetry can be diagnostic,
   * loss weighting to handle the train-set class imbalance,
   * learning rate (a few values spaced by factor 3 to 10),
   * optimizer (SGD with momentum vs Adam, the same comparison we ran on the TD4 MLP).

**Dataset handling notes.** The validation set as shipped has only 16 images, which is statistically meaningless, so we merge it back into the training set and redo a stratified 80/20 split ourselves. The test set is left untouched. Beyond accuracy we report precision, recall, F1 and a confusion matrix, because in medical screening a missed positive is not the same kind of error as a false alarm.

**Compute.** Everything is developed and trained locally in VS Code on an Apple M4 MacBook. PyTorch uses Apple's MPS backend (the GPU side of the M-series chip), which trains a small from-scratch CNN at 128x128 in about 30 seconds per epoch and a ResNet18 at 224x224 in roughly a minute per epoch. The whole project, baselines plus ablation sweep, fits comfortably in a single evening of training.

The deliverable is a single notebook telling the story end to end (data, baseline, transfer learning, ablation, comparison table, final spider chart) plus a 15-minute oral presentation.

## Tools

* **Python 3.10+**
* **NumPy** for all from-scratch work (matrix algebra, gradient computations, training loops)
* **Matplotlib** for plots and decision boundaries
* **scikit-learn** for `train_test_split`, the `make_circles` dataset, and the classification metrics on the pneumonia project
* **PyTorch** and **torchvision** for the comparison part of TD4 (autograd, optimizers) and for the CNN and transfer learning parts of the final project
* **Pillow** for image loading on the X-ray dataset
* **Jupyter** for everything that is submitted as a notebook, run locally inside **VS Code** via the Jupyter extension

## How to run

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib scikit-learn torch torchvision pillow jupyter
jupyter notebook
```

Then open whichever notebook you want to explore. The TD notebooks run on CPU in a few seconds per training loop. The pneumonia project notebook is heavier, training time depends on whether your machine has CUDA, Apple MPS, or CPU only, see the notebook for the details.

## Acknowledgments

Course taught by **Pierre Dossantos-Uzarralde**, who keeps an unusually clear path from the mathematical derivation of back-propagation all the way to the practical question of which optimizer to choose. The slide decks for the course (regression, gradient descent, perceptron, MLP) are the reference I came back to whenever a derivation was not clicking.
