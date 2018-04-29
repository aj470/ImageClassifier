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
                sums[i] = (sums[i] + alpha) / (len(images) + 2 * alpha)

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
        images = np.array([img.flat_data() for img in data.get_images()])

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
                probabilities[cls] *= np.log((params["p"] * vector) + (params["q"] * inverse)).sum()
            # get the highest guess
            guesses.append(max(probabilities.items(), key=lambda x: x[1])[0])
        return guesses
