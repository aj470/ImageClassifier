from .classifier import *


class CustomClassifier(Classifier):
    @staticmethod
    def name():
        return "custom(mira?)"

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        raise NotImplemented()

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        raise NotImplemented()