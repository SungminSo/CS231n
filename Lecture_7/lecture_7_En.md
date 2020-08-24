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



### Optimization

```python
# Vanilla Gradient Descent

whilte True:
	weights_grad = evaluate_gradient(loss_fun, data, weights)
	weights += -step_size * weights_grad # perform parameter update
```



problems with SGD(Stochastic gradient descent)

- what if loss changes quieckly in one direction and slowly in another?
  - this is referred to as the loss having a bad condition number at this point, whitch is the ratio between the largest and smallest singular values of the Hessian matrix at that point.
  - the intuitive idea is that the loss landscape kind of looks like a taco shell.
- what might SGD do on a function that looks like this?
  - if run SGD on this type of function, you might get this characteristic zigzagging behavior, where because for this type of objective function, the direction of the gradient does not align with the direction towards the minima.
  - in effect, you get very slow progress along the horizontal dimension, whitch is the less sensitive dimension, and you get this nasty zigzagging behavior across the fast-changing dimension.

- what if the loss function has a local minima or saddle point?
  - SGD will get stuch, because at this local minima, the gradient is zero because it's locally flat.
  - saddle points much more common in high dimension
  - in the regions around the saddle point, the gradient isn't zero, but the slope is very small. that means we get a very slow progress whenever our current parameter value is near a saddle point in the objective landscape.
- we often estimate the loss and estimate the gradient using a small mini batch of examples. that means we're not actually getting the true information about the gradient at every time step. instead, we're just getting some noisy estimate of the gradient at our current point.



Q: do all of these just go away if we use normal gradient descent?

A: still have the problems

- the taco shell problem of high condition numbers is still a problem with full batch gradient descent. 
- the noise, we might sometimes introduce additional noise into the network, not only due to sampling mini batches, but also due to explicit stochasticity in the network,still be a problem. 
- saddle potins, that's still a problem for full batch gradient descent because there can still be saddle points in the full objective landscape.



### SGD + Momentum

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x += learning_rate * vx
```

- have hyperparameter "rho", which corresponds to friction.



- local minina and saddle points, if we're imagining velocity in this system, then you kind of have the physical interpretation of this ball kind of rolling down the hill, picking up speed as it comes down.

- poor conditioning, the zigzagging will hopefully cancel each other out pretty fast once we're using momentum. this will effectivaly reduce the amount by which we step in the sensitive direction.
- the noise kind of gets averaged out in our gradient estimates.



Q: how does SGD momentum help with the poorly conditioned coordinate?

A: if the gradient is relatively small, and if rho is well behaved in this situation, then our velocity could actually monotonically increase up to a point where the velocity could now be larger than the actual gradient. then we might actually make faster progress along the poorly conditioned dimension.



### Nesterov Momentum

<img src="./img/momentum_update.png" />

<img src="./img/nesterov_momentum.png" />

- start at the red point. you step in the direction of where the velocity would take you. you evaluate the gradient at that point. then you go back to your original point and kind of mix together those two.

- if your velocity direction was actually a little bit wrong, it lets you incorporate gradient information from a little bit larger parts of the objective landscape.



Q: what's a good initialization for the velocity?

A: this is almost always zero. it's not even a hyperparameter. just set it to zero.



```python
dx = compute_gradient(x)
old_v = v
v = rho * v - learning_rate * dx
x += -rho * old_v + (1 + rho) * v
```

