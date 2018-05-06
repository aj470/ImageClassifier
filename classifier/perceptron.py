from .classifier import *
import random
import numpy as np
from .util import Counter

""" 
We know the following:
digit width = 28
digit height = 28
face width = 60
face height = 70
"""


class Perceptron(Classifier):
    def __init__(self, legal_labels):
        self.legalLabels = legal_labels
        self.type = "perceptron"
        self.max_iterations = 10
        self.weights = {}
        digit = True if len(legal_labels) == 10 else False

        for label in legal_labels:
            self.weights[label] = Counter()
            # Initialize Random Weights
            if digit:
                for column in range(28):  # height
                    for row in range(28):  # width
                        self.weights[label][(column, row)] = random.uniform(0, 1)
            else:  # face
                for column in range(70):  # height
                    for row in range(60):  # width
                        self.weights[label][(column, row)] = random.uniform(0, 1)

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights == weights

    def findHighWeightFeatures(self, label):
        """
		Returns a list of the 100 features with the greatest weight for some label
		"""
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        top_amount = 100
        featuresList = self.weights[label]
        featuresWeights = featuresList.sortedKeys()[
                          :top_amount]  # sort by values and then retrieve the top 100 features we want
        return featuresWeights

    def findScore(self, featureList, get_label):  # calculates score for label
        score = featureList * self.weights[get_label]
        return score

    def findMaxScore(self, mydatum):
        maxScoreLabel = self.legalLabels[0]
        maxScore = self.findScore(mydatum, self.legalLabels[0])
        i = 1
        while i < len(self.legalLabels):
            tempScore = self.findScore(mydatum, self.legalLabels[i])
            if tempScore < maxScore:
                pass
            else:
                maxScore = tempScore
                maxScoreLabel = self.legalLabels[i]
            i += 1
        return maxScoreLabel

    def train(self, training_data_set, validation_data_set):
        """Must return the final percentage accuracy achieved on validation data set."""
        training_data = [img.flat_data() for img in training_data_set.get_images()]
        training_labels = training_data_set.get_labels()
        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            indices = list(range(len(training_data)))
            while len(indices) > 0:
                datum_index = random.choice(indices)
                indices.remove(datum_index)
                training_datum = training_data[datum_index]
                training_label = training_labels[datum_index]
                temp_label = self.findMaxScore(training_datum)
                if (temp_label != training_label):
                    self.weights[training_label] = self.weights[training_label] + training_datum
                    self.weights[temp_label] = self.weights[temp_label] - training_datum

    def classify(self, data):
        """
		Classifies each datum as the label that most closely matches the prototype vector
		for that label.  See the project description for details.
    
		Recall that a datum is a util.counter... 
		"""
        guesses = []
        for datum in data:
            vectors = Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
