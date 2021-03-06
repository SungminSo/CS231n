#  Lecture 6 | Training Neural Networks I



### Mini-batch SGD

Loop:

- Sample a batch of data
- Forward prop it through the graph(network), get loss
- Backprop to calculate the gradients
- Update the parameters using the gradient



### Overview

1. One time setup
   - activation functions
   - preprocessing 
   - weight initialization
   - regularization
   - gradient checking
2. Training dynamics
   - babysitting the leaning process
   - parameter updates
   - hyperparameter optimization
3. Evaluation
   - model ensembles



### Activation Functions

- sigmoid
- tanh
- ReLU
- Leaky ReLU
- Maxout
- ELU



**Sigmoid**

<img src="./img/sigmoid_graph.png" style="zoom:67%;" />

<img src="./img/sigmoid.png" style="zoom:50%;" />

- squashes numbers to range [0, 1]
- Historically popular since they have nice interpretation as a saturating "firing rate" of a neuron
- If you get very high values as input, then output is going to be something near one. if you get very negative  values, it's going to be near zero.



3 problems:

1. Saturated neurons "kill" the gradients
   - <img src="./img/sigmoid_problem_1.png" style="zoom:67%;" />
   - what happens when x = -10? => gradient is 0
   - What happens when x = 0? => get reasonable gradient, and then it'll be fine for backprop.
   - what happens when x = 10? => gradient is 0
2. Sigmoid outputs are not zero-centered
   - if input to a neuron is always positive, it's going to be multiplied by some weight, W, and then we're going to run it through our activation function. so what can we say about the gradients on W? it's always all positive or all negative. they're always going to move in the same direction. 
   - So this gives very inefficient gradient updates.
3. exponential is a bit compute expensive
   - in the grand scheme of your network, this is usually not the main problem, because we have all these convolutions and dot products that are a lot more expensive, but this is just a minor point also to observe.



**tanh**

<img src="./img/tanh_graph.png" style="zoom:67%;" />

- squashes numbers to range [-1, 1]
- zero centered -> nice
- it still kills gradients when saturated 



**ReLU**

<img src="./img/ReLU_graph.png" style="zoom:67%;" />

<img src="./img/ReLU.png" style="zoom:50%;" />

- Does not saturate in positive region
- Very computationally efficient
- converges much faster than sigmoid/tanh in practice
- actually more biologically plausible than sigmoid



2 problems:

1. not zero-centered output
2. in the positive half of the inputs, we don't have saturation, but this is not the case of the negative half.
   - <img src="./img/ReLU_problem_1.png" style="zoom:67%;" />
   - what happens when x = -10? => gradient is 0
   - What happens when x = 0? => it undefined here, but in practice, we'll say zero
   - what happens when x = 10? => it's good. in the linear regime.

Dead ReLU

- in this bad part of the regime.
- it will never activate and never update,as compared to a n active ReLU where some of the data is going to be positive and passed through and some won't be.
- you can look at this in, as coming from several potential reasons.
  1. When you have bad initialization. then they're never going to get a data input that causes it to activate, and so they're never going to get good gradient flow coming back.
  2. when your learning rate is too high. and so this case you started off with an okay ReLU, but because you're making these huge updates, the weights jump around and then your ReLU unit in a sense, gets knocked off of the data manifold.
- so in practice,if you freeze a network that you've trained and you pass the data through, you can see it actually is much as 10 to 20% of the network is these dead ReLUs.



Q: how do you tell when the ReLU is going to be dead or not with respect to the data cloud?

A: it's whatever the weights happended to be, and where the data happens to be is where these hyperplanes fall, and so, throughout the course of training, some of your ReLUs will be in different places, with respect to the data cloud.



Q: for the sigmoid we talked about two drawbacks, and one of them was that the neurons can get saturated, but it is not the case, when all of your inputs are positive?

A: when all of your inputs are positive, they're all going to be coming in this zero plus region here, and so you can still get a saturating neuron, because you see up in this positive region, it also plateaus at one, and so when you have large positive values as input your're also going to get the zero gradient, 



So in practice people also like to initialize ReLUs with slightly positive biases, in order to increase the likelihood of it being active at initialization and to get some updates.



**Leaky ReLU**

<img src="./img/leaky_ReLU_graph.png" style="zoom:67%;" />

<img src="./img/leaky_ReLU.png" style="zoom:50%;" />

- Does not saturate
- computationally efficient
- converges much faster than sigmoid/tanh in practice!
- will not "die"



**PReLU**

<img src="./img/PReLU.png" style="zoom: 50%;" />

- the slope in the negative regime is determined through this alpha parameter, so we don't specify but we treat it as now a parameter
- so this gives it a little bit more flexibility



**ELU**

<img src="./img/ELU_graph.png" style="zoom:67%;" />

<img src="./img/ELU.png" style="zoom:50%;" />

- Has all benefits of ReLU
- closer to zero mean outputs
- negative saturation regime compared with Leaky ReLU adds some robustness to noise



problem

- computation requires exp()



Q: whether this parameter alpha is going to be specific for each neuron

A: I believe it is often specified but i actually can't remember exactly.



**Maxout**

<img src="./img/maxout.png" style="zoom:50%;" />

- does not have the basic form of dot product -> nonlinearity
- generalizes ReLU and Leaky ReLU. because just taking the max over theses two linear functions.
- operating in a linear Regime
- doesn't saturate and it doesn't die



problem

- doubles the number of parameters/neuron



### Data processing

**Step 1 : Preprocessing the data**

<img src="./img/preprocessing.png" style="zoom:67%;" />

- zero-centered: consider what happens when the input to a neuron is always positive
  - gradients on W is always all positive or all negative
- normalizing: all features are in the same range, and so that they contribute equally.



<img src="./img/PCA_Whitening.png" style="zoom:67%;" />

- in practice, you may also see PCA and Whitening of the data
- we typically just stick with the zero mean, and we don't do the normalization



one reason for this is generally with images we don't really want to take all of our input, let's say pixel values and project this onto a lower dimensional space of new kinds of features that we're dealing with. we typically just want to apply convolutional networks spatially and have our spatial structure over the original image.



Q: we do this pre-processing in a training phase, do we also do the same kind of thing in the test phase?

A: yes. in general on the training phase is where we determine our mean, and then we apply this exact same mean to the test data. so, we'll normalize by the same empirical mean from the training data.



Q: what's a channel in this case, when we are subtracting a per-channel mean?

A: this is RGB. our images are typically for example, 32 by 32 by 3. So width, height, each are 32, and our depth we have 3 channels RGB.



Q: when we're subtracting the mean image, what is the mean taken over?

A: the mean is taking over all of your training images. So you'll take all of your training images and just compute the mean of all of those.



Q: we do this for the entire training set, once before we start training, we don't do this per batch?

A: that's exactly correct. we just want to have a good sample, an empirical mean that we have. so if you take it per batch, if you're sampling reasonable batches, it should be basically getting the same values anyways for the mean, and so it's more efficient and easier just do this once at the beginning. you might not even have to really take it over the entire training data. you could also just sample enough training images to get a good estimate of your mean.



Q: does the data preprocessing solve the sigmoid problem?

A: the data preprocessing is doing zero mean. and we talked about how sigmoid, we want to have zero mean. so it does solve this for the first layer that we pass it through. but in deep network, this is not going to be sufficient.



### Weight Initialization

Q1: what happens when W=0 init is used?

- problem is all the neurons will do the same thing.
  - since your weights are zero, given an input, every neuron is going to have the same operation basically on top of your inputs. and so, since they're all going to output the same thing, they're also all going to get the same gradient. and they're all going to update in the same way. and now you're just going to get all neurons that are exactly the same, which is not you want.



Q: because the gradient also depends on our loss, won't one backprop differently compared to the other?

A: in the last layer,  like yes, you do have basically some of this, the gradient will get different loss for each specific neuron based on which class it was connected to, but if you look at all the neurons generally throughout your network, you basically have a lot of these neurons that are connected in exactly the same way. they had the same updates and it's basically going to be the problem.



-> First idea: **Small random numbers**

(Gaussian with zero mean and 1e-2 standard deviation)

<img src="./img/small_random_numbers.png" style="zoom:50%;" />

- this does work okay for small networks, but problems with deeper networks.

```python
# assume some unit gaussian 10-D input data
D = np.random.randn[1000, 500]
hidden_layer_sizes = [500]*10
nonlinearities = ['tanh']*len(hidden_layer_sizes)

act = {'relu':lambda x:np.maximum(0,x), 'tanh':lambda x:np.tanh(x)}
Hs = {}
for i in range(len(hidden_layer_sizes)):
  X = D if i == 0 else Hs[i-1] # input at this layer
  fan_in = X.shape[1]
  fan_out = hidden_layer_sizes[i]
  W = np.random.randn(fan_in, fan_out) * 0.01 # layer initialization
  
  H = np.dot(X, W) #matrix multiply
  H = act[nonlinearities[i]](H)
  Hs[i] = H # cache result on this layer
  
# look at distributions at each layer
print 'input layer had mean %f and std %f' % (np.mean(D), np.std(D))
layer_means = [np.means(H) for i, H in Hs.iteritems()]
layer_stds = [np.std(H) for i, H in Hs.iteritems()]
for i, H in Hs.iteritems():
  print 'hidden layer %d had mean %f and std %f' % (i+1, layer_means[i], layer_std[i])
  
# plot the means and standard deviations
plt.figure()
plt.subplot(121)
plt.plot(Hs.keys(), layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(Hs.keys(), layer_stds, 'or-')
plt.title('layer std')

# plot the raw distributions
plt.figure()
for i, H in Hs.iteritems():
  plt.subplot(1, len(Hs), i+1)
  plt.hist(H.ravel(), 30, range=(-1, 1))
```

<img src="./img/small_random_numbers_result.png" style="zoom:67%;" />

- the means are always around zero.
- the standard deviation shrinks and it quickly collapses to zero.
- all activations become zero



Q1: think about the backward pass. what do the gradients look like?

A: we have our input values are very small at each layer, because they've all collapsed at this near zero, and then now each layer, we have our upstream graidnet flowing down, and then in order to get the gradient on the weights  dot product were doing W times X. it's just basically going to be X, which is our inputs. so because X is small, our weights are getting a very small gradient, and they're bascially not updating.



upstream is the gradient flow from your loss, all the way back to your input. and so upstream is what came from what you've already done, flowing down into your current node.



-> Second idea: **big random numbers**

```python
# ~~~~
W = np.random.randn(fan_in, fan_out) * 1.0 # layer initialization
# ~~~~~
```

smaple from this standard gaussian, now with standard deviation 1.0 instead of 0.01

<img src="./img/big_random_numbers_result.png" style="zoom:67%;" />

- because our weights are going to be big, we're going to always be at saturated regimes of either very negative or very positive of the tanh.
- when they're saturated, that all the gradients will be zero, and our weights are not updating.



**Xavier initialization**

```python
# ~~~~
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in) # layer initialization
# ~~~~
```

1. Using tanh

<img src="./img/xavier_tanh_result.png" style="zoom:50%;" />

<img src="./img/xavier_tanh_result_graph.png" style="zoom:67%;" />

- sample from our standard gaussian, and then we're going to scale by the number of inputs that we have.
- the variance of the input to be the same as a variance of the output
- intuitively with this kind of means is that if you have a small number of inputs, then we're going to divide by the smaller number and get larget weights, because with small inputs, and you're multiplying each of these by weight, you need a larger weights to get the same larger variance at output. and kind of vice versa for if we have many inputs, then we want smaller weights in order to get the same spread at the output.



2. ReLU

<img src="./img/xavier_ReLU_result.png" style="zoom:50%;" />

<img src="./img/xavier_ReLU_result_graph.png" style="zoom:67%;" />

- when using the ReLU, nonlinearity it breaks.
- Because it's killing half of your units, it's setting approximately half of them to zero at each time, it's actually halving the variance that you get out of this.
- the distributions starts collapsing. in this case you get more and more peaked toward zero, and more units deactivated.



3. ReLU, He et al., 2015

```python
# ~~~~
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2) # layer initialization
# ~~~~
```

the way to address this with something that has been pointed out in some papers, which is that you can try to account for this with an extra, diviede by two.

<img src="./img/divide_by_two.png" style="zoom:50%;" />

<img src="./img/divide_by_two_result_graph.png" style="zoom:67%;" />

- half the neurons get killed.
- so you're kind of equivalent input has actually half this number of input, and so you just add this divided by two factor in, this works much better and the distributions are pretty good throughout all layers of the network.



<img src="./img/initialization_paper_list.png" />



### Batch Normalization

- wanting to keep activations in a gaussian range that we want.

<img src="./img/batch_normalization.png"/>

- instead of with wegith initialization, we're setting this at the start of training so that we try and get it into a good spot that we can have unit gaussians at every layer



1. compute the empirical mean and variance independently for each dimension, so each basically feature element.
2. normalize



batch normalization is usually inserted after Fully Connected or Convolutional layers, and before nonlinearity

we were multiplying by W in theses layers, which we do over and over again, then we can have this bad scaling effect with each one. and so this basically is able to undo this effect.

with convolutional layers, we want to normalize not just across all the training examples, and independently for each feature dimension, but we actually want to normalize jointly across both all the feature dimensions, all the spatial locations, that we have in our activation map as well as of the training examples.



Problem: do we necessarily want a unit gaussianinput to a tanh(nonlinearities)?

- because what this is doing is this is constraining you to the linear regime of this nonlinearity.
- but maybe a little bit of this is good, because you want to be able to control what's how much saturation that you want to have.

 

<img src="./img/batch_normalization_squash.png" />

<img src="./img/batch_normalization_gamma_beta.png" />



summary

- improves gradient flow through the network
- allows higher learning rates
- reduces the strong dependence on initialization
- acts as a form of regularization in a funny way, and slightly reduces the need for drop out



Q: is gamma and beta are learned parameters?

A: yes



Q: why do we want to learn this gamma and beta to be able to learn the identity function back?

A: because you want to give it the flexibility.what batch normalization is doing is forcing our data to become this unit gaussian, but even though in general this is a good idea, it's not always that this is exactly the best thing to do. we saw in particular for something like a tanh, you might want to control some degree of saturation that you have. so what this does is it gives you the flexivility of doing this esact like unit gaussian normalization if it wants to, but also learning that maybe in this particular part of the network slightly scaled or shifted.



Q: for things like reinforcement learning, you might have a really small bach size. how do you deal with this?

A: in practice, batch normalization has been used a lot for like for standard convolutional neural networks, and there's actually papers on how do we want to do normalization for different kinds of recurrent networks. and so there's different considerations that you might want to think of there.



Q: if we force the inputs to be gaussian, do we lose the structure?

A: no, in a sense that you can think of if you had all your features distributed as a gaussian for example, even if you were just doing data pre-processing, this gaussian is not losing you any structure. it's just shifting and scaling your data into a regime.



Q: are we normalizing the weight so that they become gaussian?

A: we're normalizing the inputs to each layer, so we're not changing the weights in this process. 



Q: once we subtract by the mean and divide by the standard deviation, does this become gaussian?

A: yes. 



Q: if we're going to be doing the shift and scale, and learning these is the batch normalization redundant, because you could recover the identity mapping?

A: the network learns that identity mapping is alwyas the best, and it learns these parameters, there would be no point for batch normalization, but in practice this doesn't happen. so in practice, we will learn this gamma and beta.that's not the same as a identity mapping. it will shift and scale by some amount, but not the amount that's going to give you an identity mapping. 



we don't re-compute mean and std at test time. instead fixed empirical mean of activations during training is used.



### Babysitting the Learning Process

- how do we monitor training
- how do we adjust hypterparameters as we go to get a good result

1. Preprocess the data
2. Choose the architecture
3. Start with small regularization and find learning rate that makes the loss go down



Q1: even though our loss with barely changing, the training and the validation accuracy jumped up to 20% verfy quickly. why this might be the case?

​	-> the probabilities are still pretty diffuse, so our loss term is still pretty similar. but when we shift all of thses probabilities slightly in the right direction, the accuracy all of a sudden can jump, because we're taking the maximum correct value.



Q2: when you learning with quite big learning rate, you have NaNs. what's the meaning of NaNs in this case?

​	-> NaN almost alwyas means high learning rate, and your cost exploded.



### Hyperparameter Optimization

- the strategy that we're going to use is for any hyperparameter for example learning rate, is to do cross-validation.



Cross-valication

- train on your training set, and then evaluating on a validation set.



if you want to make sure that your hyperparameter's range kind of has the good values somewhere in the middle, or somewhere you get a sense that you've hit, you've explored your range fully.



Random Search vs Grid Search

- we can sample for a fixed set of combinations, a fixed set of values for each hyperparameter.
- but in pracitce it's actually better to sample from a random layout
- <img src="./img/sample_search.png" />



Monitor loss curve

<img src="./img/monitor_loss_curves.png" />

<img src="./img/result_of_bad_initialization.png" />

