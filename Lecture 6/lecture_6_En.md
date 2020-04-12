# Lecture 3 | Loss Functions and Optimization



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
- converges much faster than sigmoid/tanh in practivce
- actually more biologically plausible thant sigmoid



2 problems:

1. not zero-cetnered output
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
- so if in practice,if you freeze a network that you've trained and you pass the data through, you can see it actually is much as 10 to 20% of the network is these dead ReLUs.



Q: how do you tell when the ReLU is going to be dead or not with respect to the data cloud?

A: it's whatever the weights happended to be, and where the data happens to be is where these hyperplanes fall, and so, throughout the course of training, some of your ReLUs will be in different places, with respect to the data cloud.



Q: for the sigmoid we talked about two drawbacks, and one of them was that the neurons can get saturated, but it is not the case, when all of your inputs are positive?

A: so when all of your inputs are positive, they're all going to be coming in this zero plus region here, and so you can still get a saturating neuron, because you see up in this positive region, it also plateaus at one, and so when it's when you have large positive values as input your're also going to get the zero gradient, 



So in practice people also like to initialize ReLUs with slightly positive biases, in order to increase the likelihood of it being active at initialization and to get some updates.



**Leaky ReLU**

<img src="./img/leaky_ReLU_graph.png" style="zoom:67%;" />

<img src="./img/leaky_ReLU.png" style="zoom:50%;" />

- Does not saturate
- computationally efficient
- converges much faster than sigmoid/tanh in practice!
- will not "die"



