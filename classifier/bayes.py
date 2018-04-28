import numpy as np

from .classifier import *


class BayesClassifier(Classifier):
    def __init__(self, legal_labels):
        """Bayes classifier for Bernoulli distributions"""
        super().__init__(legal_labels)
        self.priors = {}
        self.class_parameters = {}

    @staticmethod
    def name():
        return "bayes"

    def _separate(self, dataset):
        # dict mapping classes to lists of images
        classes = {}
        for data, cls in dataset:
            if cls not in classes:
                # if class is not already in dict, add it
                classes[cls] = []
            classes[cls].append(data)
        return classes

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        classes = self._separate(training_data.get_labeled_images())
        total_size = len(training_data)
        alpha = 1
        # iterate through each possible class
        for cls in classes:
            images = classes[cls]               # images assigned to each class
            flat_images = tuple(map(lambda x: x.flat_data(), images))  # flat images
            dimension = len(flat_images[0])     # number of features
            sums = [0]*dimension                # array of feature sums

            for img in flat_images:
                # for each image, go through each feature
                for i in range(dimension):
                    # increment counter of this feature if feature is non-zero
                    if img[i] > 0:
                        sums[i] += 1

            # go through each feature and convert it to a probability
            for i in range(dimension):
                # => number of occurrences of feature / maximum possible number of occurrences (+ alpha adjustments)
                sums[i] = (sums[i] + alpha) / (len(images) + alpha)

            # convert to array
            sums = np.array(sums)
            # store p and q arrays:
            # probability of feature being present in this class, and probability of it not being present
            self.class_parameters[cls] = {"p": sums, "q": 1 - sums}
            # set prior - number of occurrences of this class in relation to size of dataset
            self.priors[cls] = len(images) / total_size

        return self.validate(validation_data)

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        guesses = []
        for image, label in data.get_labeled_images():
            # get flat image == feature vector
            vector = image.flat_data()
            probabilities = {}
            # convert it to an np array
            vector = np.array(vector)
            # calculate inverse => 0's become 1's and 1's become 0's
            inverse = 1 - vector
            for cls, params in self.class_parameters.items():
                # uniform prior works slightly better than actual prior => dataset is roughly uniformly distributed
                probabilities[cls] = 1
                # calculate P(f1 | C) * P(f2 | C) * P(f3 | C) ... * P(fn | C)
                # by calculating p * vector (probability of '1' features occurring) + q * inverse (probability of '0' features occurring)
                probabilities[cls] *= np.prod((params["p"] * vector) + (params["q"] * inverse))
            # get the highest guess
            guesses.append(max(probabilities.items(), key=lambda x: x[1])[0])
        return guesses