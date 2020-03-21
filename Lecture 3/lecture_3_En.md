# Lecture 3 | Loss Functions and Optimization



TODO

1. Define a **loss function** that quantifies our unhappiness with the scores across the training data.
2. Come up with a way of effieiently finding the parameters that minimize the loss function(**optimization**)



Example

- Suppose: 3 training examples, 3 classes.
- With some W the scores f(x, W) = Wx
- <img src="./img/loss_function_example.png" />



**loss function** tells how good our current classifier is

<img src="./img/loss_function_dataset_numerical_expression.png" style="zoom:50%;" />

<img src="./img/loss_function_numerical_expression.png" style="zoom:50%;" />



### Multiclass SVM loss

<img src="./img/SVM_numerical_expression.png" />

- perform a sum over all of the categories, Y, except for the true category, Y_i
- so, sum over all the incorrect categories and then we're going to compare the score of the correct category, and the score of the incorrect category.
- now if the score for the correct category is greater than the score of the incorrect category(greater than the incorrect score by some safety margin that we set to one), it's means that the score for the true category is much larger than any of the false categories, then we'll get a loss of zero.
- this kind of like "if then" statement
- it is often referred to as some type of a hinge loss
  - <img src="./img/SVM_graph.png" />



Q: In terms of notation, what is S and what is S_y_i in particular?

A: the S are the predicted scores for the classes that are coming out of the classifier. and that y_i is the category of the ground truth label for the example which is some integer. so S_y_i corresponds to the score of the true class for the i-th example in the training set.



Q: What exactly is this computing here?

A: in some sense, what this loss is saying is that we are happy if the true score is much higher than all the other scores. It needs to be higher than all the other scores by some safety margin. and if the true score is not high enough, greater than any or the other scores, then we will incur some loss and that would be bad.1

