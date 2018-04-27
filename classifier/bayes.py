from .classifier import *
from sklearn.naive_bayes import MultinomialNB as nb
import numpy as np


class BayesClassifier(Classifier):
    def __init__(self, legal_labels):
        super().__init__(legal_labels)
        self.model = nb(20, fit_prior=False)

    @staticmethod
    def name():
        return "bayes"

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        images, labels = training_data.get_images(), training_data.get_labels()
        images = [image.flat_data() for image in images]
        images = np.array(images)
        labels = np.array(labels)
        self.model.fit(images, labels)

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        images = [image.flat_data() for image in data.get_images()]
        images = np.array(images)
        return list(self.model.predict(images))