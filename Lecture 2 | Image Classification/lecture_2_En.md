# Lecture 2 | Image Classification



Image Classification: A core task in Computer Vision



the problem: Semantic Gap

- the computer is representing the image as gigantic grid of numbers.
- for example, the image might be something like 800 by 600 pixels, each pixels is represented by three numbers, giving the red, green, and blue values for that pixel.



Challenges: Viewpoint variation

- when change the picture in very small, subtle ways that will cause this pixel grid to change entirely.



Challenges: Illumination

- for example, whether the cat is apperaing in this very dark, moody scene, or very bright, sunlit scene, it's still a cat. And our algorithms need to be robust to that



Challenges: Deformation

- for example, cats can really assume a lot of different, varied poses and position. And our algorithms should be robust to these different kinds of transforms.



Challenges: Occlusion

- if you might only see part of an object, you can realize that object. And our algorithms also must be robust to.



Challenges: Background Clutter

Challenges: Intraclass variation

- for example, one notion of cat-ness, actually spans a  lot of different visual appearances. and cats can come in different shapes and sizes, colors and ages. And our algorithms ,again, needs to work and handle all these different variations. 



when we're trying to recognize objects, there's no really clear, explicit algorithm. unlike sorting a list of numbers.



Attempts have been made

- Edges are pretty import when it comes to visual recognition.
- so one thing we might try to do is compute the edges of this image and then go in and try to categorize all the different corners and boundaries and write explicit set of rules for recognizing object.
- But this turns out not to work very well. because
  1. it's super brittle
  2. if you want to start over for another object category, you need to start all over again.



Data-Driven Approach

1. Collect a dataset of images and labels
2. Use Machine Learning to train a classifier
3. Evaluate the classifier on new images



```python
def train(images, labels):
	# Machine learning!
  return model
```

 ```python
def predict(model, test_images):
	# Use model to predict labels
	return test_labels
 ```



 





