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
        classes = {}
        for data, cls in dataset:
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(data)
        return classes

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        classes = self._separate(training_data.get_labeled_images())
        total_size = len(training_data)
        alpha = 1
        for cls in classes:
            images = classes[cls]
            flat_images = tuple(map(lambda x: x.flat_data(), images))
            dimension = len(flat_images[0])
            sums = [0]*dimension

            for img in flat_images:
                for i in range(dimension):
                    if img[i] > 0:
                        sums[i] += 1

            for i in range(dimension):
                sums[i] = (sums[i] + alpha) / (len(images) + alpha)

            sums = np.array(sums)
            self.class_parameters[cls] = {"p": sums, "q": 1 - sums}
            self.priors[cls] = len(images) / total_size

        return self.validate(validation_data)

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        guesses = []
        for image, label in data.get_labeled_images():
            vector = image.flat_data()
            probabilities = {}

            vector = np.array(vector)
            inverse = 1 - vector
            for cls, params in self.class_parameters.items():
                probabilities[cls] = self.priors[cls]
                probabilities[cls] *= np.dot(params["p"], vector) * np.dot(params["q"], inverse)

            guesses.append(max(probabilities.items(), key=lambda x: x[1])[0])
        return guesses