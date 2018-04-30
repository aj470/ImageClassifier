from .classifier import *
import random
import numpy as np
import util
PRINT = true

""" 
We know the following:
digit width = 28
digit height = 28
face width = 60
face height = 70
"""

class Perceptron(Classifier):
	def __init__(self, legal_labels, max_iterations):
		self.legal_labels = legal_labels
		self.type = "perceptron"
		self.max_iterations = max_iterations
		self.weights = {} 
		for label in legal_labels:
			self.weights[label].util.Counter()
		#Initialize Random Weights 
		if(digit):
			for column in range(28):  #height 
				for row in range(28): #width
					self.weights[label][(column,row)] = random.uniform(0,1)
		else: #face
			for column in range(70):  #height
				for row in range(60): #width
					self.weights[label][(column,row)] = random.uniform(0,1)
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
		featuresList = self.weights[label]
		featuresWeights = featuresList.sortedKeys()[:100] #sort by values and then retrieve the top 100 features we want
		return featuresWeights		
      
    def train(self, training_data, training_labels, validation_data, validation_labels):
        """Must return the final percentage accuracy achieved on validation data set."""
        raise NotImplemented()

	def classify(self, data ):
		"""
		Classifies each datum as the label that most closely matches the prototype vector
		for that label.  See the project description for details.
    
		Recall that a datum is a util.counter... 
		"""
		guesses = []
		for datum in data:
			vectors = util.Counter()
			for l in self.legalLabels:
				vectors[l] = self.weights[l] * datum
			guesses.append(vectors.argMax())
		return guesses