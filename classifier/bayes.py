import numpy as np

from .classifier import *


class BayesClassifier(Classifier):
    def __init__(self, legal_labels):
        """Bayes classifier for Bernoulli distributions"""
        super().__init__(legal_labels)
        self.log_params = None
        self.inverse_log_params = None
        self.feature_sums = None
        self.class_sums = None
        self.classes = None
        self.priors = None

    @staticmethod
    def name():
        return "bayes"

    def _count(self, dataset):
        # dict mapping classes to lists of images
        classes = {}
        for img, cls in dataset:
            if cls not in classes:
                # if class is not already in dict, add it
                classes[cls] = []
            classes[cls].append(img.flat_data())

        # one dimensional array of classes
        self.classes = np.array(list(classes.keys()))
        # class_images is a list -- each element is the set of images belonging to a corresponding class
        # it would be a matrix, but classes may not (most likely don't) have equal number of images.
        class_images = [np.array([image for image in classes[cls]]) for cls in self.classes]
        # sum features present per class
        self.feature_sums = np.array([imgs.sum(axis=0) for imgs in class_images])
        # number of items belonging to each class
        self.class_sums = np.array([len(arr) for arr in classes.values()])
        # calculate priors (% of the time a class occurred in dataset)
        self.priors = np.log(self.class_sums) - np.log(self.class_sums.sum())

    def _smooth(self, alpha):
        # feature counts is not a 2D array, each row representing a class, each element representing the number of
        # occurrences of that features in said class throughout all training data
        feature_counts = np.log(self.feature_sums + alpha)
        # number of samples belonging to each class
        class_counts = np.log(self.class_sums + len(self.class_sums) * alpha).reshape(-1, 1)
        # compute (# of feature occurrences) / (# of possible occurrences) to get probability of feature per class
        self.log_params = feature_counts - class_counts
        # q array = (1 - p)
        self.inverse_log_params = np.log(1 - np.exp(self.log_params))

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        self._count(training_data.get_labeled_images())
        # alpha = 1
        # delta = 5
        # iterations = 10
        # prev_accuracy = 0
        # best_accuracy = 0
        best_alpha = 1
        # for i in range(iterations):
        #    print(alpha)
        #    self._smooth(alpha)
        #    accuracy = self.validate(validation_data)
        #    if accuracy > best_accuracy:
        #        best_accuracy = accuracy
        #        best_alpha = alpha
        #    elif accuracy < prev_accuracy:
        #        delta = - delta / 2
        #    prev_accuracy = accuracy
        #    alpha = max(1, alpha + delta)
        # print("Using alpha: ", best_alpha)
        self._smooth(best_alpha)
        return self.validate(validation_data)

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        images = np.array([img.flat_data() for img in data.get_images()])

        guesses = []
        for image in images:
            inverse_img = 1 - image
            # image * log_params calculates P(f[i] = 1 | C)
            # inverse_img * inverse_log_params calculates P(f[i] = 0 | C)
            # adding these two arrays results in P(f[i] | C) for all i in f
            class_probabilities = inverse_img * self.inverse_log_params + image * self.log_params
            # The sum of P(f[i:n] | C) is equal to the probability that x belongs to C
            # P(C | f1...fn) ~~ P(f1...fn | C) * P(C)
            # P(f1...fn | C) = P(f1 | C) * P(f2 | C) * P(f3 | C) * ... * P(fn | C)
            # since they are log probabilities, we can simply add them
            class_probabilities = class_probabilities.sum(axis=1)
            # get max probability, and add the corresponding class to the list of guesses
            guesses.append(self.classes[class_probabilities.argmax()])
        return guesses
