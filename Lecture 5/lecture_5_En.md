# Lecture 5 | Convolutional Neural Networks



### Convolutional Neural Networks

<img src="./img/convolutional_neural_network.png" />



### A bit of history

- 1957, Frank Rosenblatt
  - developed the Mark 1 perceptron machine, which was the first implementation of an algorithm called the perceptron, which had sort of the similar idea of getting score functions.
  - but the outputs are going to be either one or a zero.
  - have an update rule for weights W, which also look kind of similar to the type of update rule that we're also seeing in backprop, but in this case there was no principled backpropagation technique. just sort of took the weights and adjusted them in the direction towards the target that we wanted.
- 1960, Widrow and Hoff
  - developed Adaline and Madaline, which was the first time to start to stack these linear layers into multilayer perceptron networks
  - but still didn't have backprop or any sort of principled way to train this.
- 1986, Rumelhart
  - first time to introduce backpropagation.
- 2006, Geoff Hinton and Ruslan Salakhutdinov
  - basically showed that we could train a deep neural network and show that we could do this effectively.
  - but it was still not quite the sort of modern iteration of neural networks.
  - it required really careful initialization in order to be able to do backprop, so they would have first pre-training stage, where you model each hidden layer through this kind of, a restricted Boltzmann machine, and get some initialized weights by training each of these layers iteratively.
- 2012, Geoff Hinton's lab
  - acoustric modeling and speech recognition.
  - for image recognition, from Alex Krizhevsky in Geoff Hinton's lab, which introduced the first convolutional neural network architecture that was able to do get really stron results on ImageNet classification.



### Fully Connected Layer

<img src="./img/fully_connected_layer.png" />



### Convolution Layer

- Convolve the filter with the image
  - small filter, for example 5x5x3 filter, slide it over the image, for example 32x32x3 size, spatially and compute dot products at every spatial location.
- Filters : always extend the full depth of the input volume, and just a smaller spatial area of inpute image



Q: when we do the dot product do we turn the 5x5x3 into one vector?

A: yes, in essence that's what you're doing.



Q: any intuition for why this is a W transpose?

A: this is just not really. this is just the notation that we have here to make the math work out as a dot product. 

