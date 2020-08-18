# Lecture 7 | Training Neural Networks ||



### Review

- Activation Functions
  - ReLU
  - tanh
  - Maxout
- Weight Initialization
  - too small: Activations go to zero, gradients also zero, no learning
  - too big: Activations saturate, gradients zero, no learning
- Data Preprocessing
  - Zero-centered data
  - Normalized data: classificatio loss very sensitive to changes in weight matrix, hard to optimize
- Batch Normalization
  - Add this additional layer inside our networks to just force all of the intermediate activations to be zero mean and unit variance.



Q: how many hyperparameters do we typically search at a time?

A: it kind of depends on the exact model and the exact arcitecture, but because the number of possibilities is exponential in the number of hyperparameters you can't really test too many at a time. but generally, I try not to do this over more four at a time at a most.



Q: how often does it happen when you change one hyperparameter, then the other, the optimal values of the other hyperparameters change?

A: that does happen sometimes, although for learning rates, that's typically less of a problem.



Q: what's wrong with having a small learning rate and increasing the number of epochs?

A: it might take a very long time.



Q: for a low learning rate, are you more likely to be stuck in local optima?

A: I think that makes some intuitive sense, but in practice, that seems not to be much of a problem.



### Overview

- Fancier optimization
- Regularization
- Transfer Learning



