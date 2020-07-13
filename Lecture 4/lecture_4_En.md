# Lecture 4 | Introduction to Neural Networks



TODO

- how to compute the analytic gradient for arbitrarily complex functions

 

### Gradient descent

1. Numerical gradient
   - pros:
     - easy to write
   - cons:
     - slow
     - approximate
2. Analytic gradient
   - pros:
     - fast
     - exact
   - cons:
     - error-prone



### Computational graphs

<img src="./img/computational_graph.png" style="zoom:67%;" />

- the advantage is that once we can express a function using a computational graph

  then we can use a technique that we call backpropagation whitch is going to recursively use the chain rule in order to compute the graident with respect to every variable in the computational graph



### Backpropagation

<img src="./img/backprop_example.png" style="zoom:67%;" />

- forward : (-2 + 5) * (-4) = -12

- <img src="./img/backprop_example_1.png" style="zoom:50%;" />

  

- bakcward :

   <img src="./img/backprop_example_result.png" style="zoom:50%;" />



Q: Can you go back and explain why more in the last slide was different than planning the first part of it using just normal calculus?

A: we can do calculate in the example because it's simple, but we'll see examples later on where once this becomes a really complicated expression. you don't want to have to use calculus to derive the gradient for something for a super-complicated expression, and instead, if you use  this formalism and you break it down into these computational nodes, then you can only ever work with gradients of very simple computations.



during backprop, we'll start from the back of the graph.

when we reach each node, at each node we have the upstream gradients coming back with respect to the immediate output of the node. so by the time we reach this node in backprop, we've already computed the gradient of our final loss L, with respect to z.

we have from the chain rule, that the gradient of this loss function with respect to x is going to be the gradient with respect to z times, compounded by local gradient of z with respect to x.

so in chain rule we always take this upstream gradient coming down, and we multiply it by the local gradient in order to get the gradient with respect to the input



Q: whether this only works because we're working with the current values of the function that we plug in but we can write an expression for this still in terms of the variables?

A: So we'll see that gradient of L with respect to z is going to be some expression, and gradient of z with respect to x is going to be another expression. but we plug in the values of these numbers at the time in order to get the value of the gradient with respect to x. so as you said, basically this it going to be just a number coming down, and then we just multiply it with the expression that we have for the local gradient.



<img src="./img/backprop_another_example.png" style="zoom:80%;" />

- we're going to start at the very end of the graph, and so here again the gradient of the output with respect to the last variable is just one, it's just trivial.
- once we had these expressions for local gradients, all we did was plug in the values for each of these that we have and use the chain rule to numerically multiply this all the way backwards and get the gradients with respect to all of the variables.



Q: This is a question on the graph itself, is there a reason that the first two multiplication nodes and the weights are not connected to a single addition node?

A: they could also be connected into a single addition node, you can do that if you want. in this case i just wrote this out into as simple as possible, where each node only had up to two inputs.



### Patterns in backward flow

"add" gate : gradient distributor

	- pass the exact same thing to both of the branches that were connected.

"max" gate : gradient router

- just take the gradient and route it to one of the branches.

"mul" gate : gradient switcher

- take the upstream gradient and scale it by the value of the other branch.









