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



### AdaGrad

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

- during the course of the optimization, you're going to keep a running estimate or a running sum of all the squared gradients that you see during training.
- Q1: what does this kind of scaling do in this situation where we have a very high condition number?
  - if we have two coordinates, one that always has a very high gradient and one that always has a very small gradient, then as we add the sum of the squares of the small gradient, we're going to be dividing by a small number, so we'll accelerate movement along the one dimension.
  - then along the other dimension, where the gradients tend to be very large, then we'll be dividing  by a large number, so we'll kind of slow down our progress along the wiggling dimension.
- Q2:  what happens with AdaGrad over the course of training
  - the steps actually get smaller and smaller because we just continue updating this estimate of the squared gradients over time, so this estimate just grows monotonically over the course of training.
  - in the non-convex case, that's a little bit problematic, because as you come towards a saddle point, you might get stuck with AdaGrad, and then you kind of no longer make any progress.



### RMSProp

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```



### Adam

```python
# almost (not full form)
first_moment = 0
second_moment = 0
while True:
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```

- first_moment -> role of momentum
- second_moment -> role of AdaGrad/RMSProp
- Q1: what happens at the first time step?
  - we've initialized our second moment with zero. after one update of the second moment, second moment still very close to zero. so then when we're making our update step here and we divide by our second moment now we're dividing by a very small number. so we're making a very large step at the beginning.



Q: the first moment at the first time step is also very small, then you're multiplying by small and you're dividing by square root of small squared, so what's going to happen? they might cancel each other out.

A: that's true. sometimes these cancel each other out and ou're okay. but sometimes this ends up in taking very large steps right at the beginning.



Q: what is this 10 to the minus seven term in the last equation?

A: the idea is that we're dividing by something. we want to make sure we're not dividing by zero, so we always add a small positive constant to the denominator, just to make sure we're not dividing by zero. 



```python
# full form
first_moment = 0
second_moment = 0
while True:
    dx = compute_gradient(x)
    # Momentum
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    # AdaGrad/RMSProp
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    # Bias Correction
    first_unbias = first_moment / (1 - beta1 **t)
    second_unbias = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```

- update our first and second moments, we create an unbiased estimate of those first and second moments by incorporating the current time step, t.



Q: what does Adam not fix?

A: they still take a long time to train. 



### Learning Rate

<img src="./img/learning_rate.png" />

- Q1: Which one of these learning rates is best to use?
  - we don't actually have to stick with one learning rate throughout the course of training.
  - => learning rate decay over time!
    - step decay
    - exponential decay
    - 1/t decay 
  - learning rate decay is a little bit more common with SGD momentum, and a little bit less common with Adam.
  - learning rate dacy is kind of a second-order hyperparameter. you typically should not optimize over this thing from the start. try with no decay, see what happens. then kind of eyeball the loss curve and see where you think you might need decay.



we don't really care about training error that much.

instead, we really care about our performance on unseen data.



### Model Ensembles

1. Train multiple independent models
2. At test time average their results



Q: it's bad when there's a large gap between error that means you're overfitting, but if therer's no gap, then is that also matbe bad? Do we actually want some small, optimal gap between the two?

A: we don't really care about the gap. what we really care about is maximizing the performance on the validation set. so what tends to happen is that if you don't see a gap, then you could have improved your absolute performance, in many cases, by overfitting a little bit more. 



Q: are hyperparameters the same for the ensemble?

A: somethimes they're not. you might want to try different sizes of the model, different learning rates, different regularization strategies and ensemble across these different things.



Tips and Tricks

- instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time (Polyak averaging)

- ```python
  while True:
      data_batch = dataset.sample_data_batch()
      loss = network.forward(data_batch)
      dx = network.backward()
      x += - learning_rate * dx
      x_test = 0.995*x_test + 0.005*x
  ```



### Regularization

- where we add something to our model to prevent it from fitting the training data to well in the attempts to make it perform better on unseen data.



in common use:

	- L2 regularization
	- L1 regularization
	- Elastic net(L1 + L2)



Dropout

- in each forward pass, randomly set some neurons to zero
- probability of dropping is a hyperparameter
- 0.5 is common



Q: what are we setting to zero?

A: it's the activations.



Q: which layers do you do this on?

A: it's more common in fully connected layers, but you sometimes see this in convolutional layers, as well. 



```python
p = 0.5 # probability of keeping a unit active.

def train_step(X):
    """ X contains the data """
    
    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p #first dropout mask
    H1 *= U1 # drop!
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p # second dropout mask
    H2 *= U2 # drop!
    out = np.dot(W3, H2) + b3
    
    
def predic(X):
    # ensembled forward pass
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
    out = np.dot(W3, H2) + b3
```



Q: what happens to the gradient during training with dropout?

A: we only end up propagating the gradients throught the nodes that were not dropped. this has the consequence that when you're training with dropout, typically training takes longer because at each step, you're only updating some subparts of the network. 



### Data Augmentation

- transform the image in some way during training such that the label is preserved.
- horizontal flips
- color jitter
- etc



### DropConnect

- related idea to dropout, but rather than zeroing out the activations at  every forward pass, instead we randomly zero out some of the values of the weight matrix instead.



### Fractional Max Pooling

- every time we have our pooling layer, we're going to randomize exactly the pool that the regions over which we pool.



Q: do you usually use more than one regularization method?

A: in many cases, batch normalization alone tends to be enough, but then sometimes if batch normalization alone is not enough, then you can consider adding dropout or other thing once you see your network overfitting.



### Transfoe Learning 

- one problem with overfitting is sometimes you overfit 'cause you don't have enough data.
- transfer learning kind of busts this myth that you don't need a huge amount of data in order to train a CNN.
- <img src="./img/transfer_learning.png" />





