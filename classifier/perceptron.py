from .classifier import *
import numpy as np
import util
PRINT = true

class Perceptron(Classifier):
	def __init__(self, legal_labels):
		self.legal_labels = legal_labels
		self.type = "perceptron"
		self.weights = {}
		for label in legal_labels:
			self.weights[label].util.Counter()
    @staticmethod
    def name():
        return "perceptron"
	def setWeights(self, weights):
		assert len(weights) == len(self.legalLabels);
		self.weights == weights;
		
	def findHighWeightFeatures(self, label):
		"""
		Returns a list of the 100 features with the greatest weight for some label
		"""
		featuresWeights = []

		"*** YOUR CODE HERE ***"
		util.raiseNotDefined()

		return featuresWeights		
      
    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        raise NotImplemented()

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        raise NotImplemented()