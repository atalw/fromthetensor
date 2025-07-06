# From the Tensor to the Transformer 

This repository contains a series of mini-projects that build up the core components of modern deep learning frameworks from scratch all they way up to a full-blown transformer model.

## Section 1: The Building Blocks: Autograd Engine 

- **Building a Scalar Autograd Engine** (150 lines) -- We begin by building a scalar autograd engine inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Create a `Value` object that wraps a single number and overloads Python's arithmetic operators (+, *) to build a computational graph on the fly. Each Value object knows the children it was created from and has a `.backward()` method that automatically computes the gradient of the output with respect to every node in the graph using the chain rule. This is the fundamental concept behind backpropagation.

## Section 2: Making it Real: A Tensor Library

- **Coding a Tensor Library** (500 lines) -- Moving from scalars to multi-dimensional arrays, this project builds a fully-featured Tensor library. You'll create a powerful Tensor object ([tensor.py](https://github.com/atalw/fromthetensor/blob/main/lilgrad/lilgrad/tensor.py)) with common tensor operations like `expand`, `permute`, `pad`, `cat`, `squeeze`, `slice`, `conv2d`, `batchnorm`, `layernorm`, etc. You can do [tensor puzzles](https://github.com/srush/Tensor-Puzzles) by Sasha Rush to gain an intuition of how tensors [work](https://github.com/atalw/fromthetensor/blob/main/maths/tensor_puzzles.ipynb).
- **Backpropagation with Operations** (300 lines) -- With the `Tensor` object defined, you'll implement a suite of operations in [ops.py](https://github.com/atalw/fromthetensor/blob/main/lilgrad/lilgrad/ops.py), from basic arithmetic to neural network-specific functions like `Conv2D` and `MaxPool`. Each operation is a [Function](https://github.com/atalw/fromthetensor/blob/main/lilgrad/lilgrad/function.py) with a defined forward and backward pass, creating your own mini-PyTorch or mini-tinygrad.
- **Optimizers** (100 lines) -- To make our library capable of training neural networks, you'll implement standard optimization algorithms like `SGD` and `Adam` in [optim.py](https://github.com/atalw/fromthetensor/blob/main/lilgrad/lilgrad/nn/optim.py). These optimizers take a set of tensors and update them according to their computed gradients, completing our from-scratch deep learning framework.

## Section 3: Building a Neural Network

- **Implementing Neural Network Layers** (50 lines) --  We'll use our new tensor library to build the fundamental layers of a [neural network](https://github.com/atalw/fromthetensor/blob/main/lilgrad/lilgrad/nn/__init__.py): Linear layers, ReLU, Softmax, and more.
- **Training your First Neural Network** (50 lines) -- With the layers in place, you'll write a simple training loop to train a neural network on a real dataset (like [MNIST](https://github.com/atalw/fromthetensor/blob/main/lilgrad/examples/mnist.py)).

## Section 4: Classic Models and Architectures

- **Boston Housing Price Prediction (Linear Regression)** (80 lines) -- A classic introductory project. You'll implement a simple linear regression model from scratch to predict housing prices, solidifying your understanding of loss functions and gradient descent.
- **CIFAR Image Classification** (80 lines) -- In this [project](https://github.com/atalw/fromthetensor/blob/main/cifar/train.py), you'll build a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. You'll implement convolution and pooling layers to see how they excel at image-based tasks. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)]
- **DeepChess** (500 lines) -- Build a complete chess engine. You'll start by creating a `pos2vec` model that learns to represent chess positions as vectors. Then, you'll use a siamese network to compare board positions and a distilled model to create a smaller, faster version of your engine. [[Paper](https://arxiv.org/abs/1711.09667)]
- **Recurrent Neural Networks (RNNs)**  (100 lines) -- Let's dive into sequence data. In this project, you will build a simple classification model that can correctly determine the nationality of a person given their name. [[Blog](https://jaketae.github.io/study/pytorch-rnn/)]
- **GRU & LSTMs** (200 lines) -- As an upgrade to our RNN, this [project](https://github.com/atalw/fromthetensor/blob/main/gru_lstm/main.py) has you implement Gated Recurrent Units (GRUs) and Long Short-Term Memory (LSTM) networks. You'll see firsthand how gating mechanisms can help capture long-range dependencies in sequences and overcome the vanishing gradient problem. You'll build model that can generate polyphonic music. [[Paper](https://arxiv.org/pdf/1412.3555)]

## Section 5: The Attention Revolution

- **Implementing a Transformer** (400 lines) -- You'll construct each component of a transformer step-by-step: the embedding layer, positional encoding, the multi-head self-attention mechanism, feed-forward networks, and layer normalization. Part one of this [project](https://github.com/atalw/fromthetensor/tree/main/transformer) will just be sentiment analysis of IMDB reviews. For part two, you'll extend this encoder-only mechanism to an encoder-decoder architecture and generate reviews yourself. [[Paper](https://arxiv.org/abs/1706.03762)]
