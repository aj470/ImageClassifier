ImageClassifier

This project is an image classifier program that uses 3 different kinds of machine learning algorithms to classify
images that are stored in text files. Currently it can classify either single digit hand written numbers, or determine
whether or not there is a face in the image. The three algorithms used are:
    - Naive Bayes classifier - multinomial implementation
    - Single layer multiclass perceptron
    - K nearest neighbors

All three algorithms are implemented using Numpy. Accuracy on the Bayes classifier is approximately 77% of digits and
90% on faces. The perceptron is able to achieve 80% accuracy on digits, while maintaining the same 90% on faces.
KNN performs the best out of the three algorithms for classifying digits at 87% accuracy, but is only able to achieve
75% accuracy when identifying faces.

These classifiers would likely be able to achieve higher accuracy if any sort of filtering or feature extraction
was performed on the images before processing. At the moment, these are the accuracies just reading the raw image data.

usage: main.py [-h] [-f] [-a {1,2,3,4,5}] [-d {face,digit}] [-c]

Example usage: ./main.py -a 1 -d face
               ./main.py -f

CS440 Project 2 -- Image Classification

optional arguments:
  -h, --help            show this help message and exit
  -f, --full-eval       Run all algorithms on both faces and digits
  -a {1,2,3,4,5}, --algorithm {1,2,3,4,5}
                        1. Perceptron, 2. Naive Bayes Classifier, 3. K-nearest neighbors
  -d {face,digit}, --data {face,digit}
                        whether to evaluate faces or digits
  -c, --csv             Output CSV data

