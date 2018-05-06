# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from .classifier import *
from .util import Counter
import random
import numpy as np

""" 
We know the following:
digit width = 28
digit height = 28
face width = 60
face height = 70
"""

class Perceptron(Classifier):
    """
    Perceptron classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
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
                        
    @staticmethod
    def name():
        return "perceptron"
        
    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def _separate(self, dataset):
        classes = {}
        for data, cls in dataset:
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(data)
        return classes

    def train(self, training_data, validation_data):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        classes = self._separate(training_data.get_labeled_images())
        training_data = {}
        k = 0
        training_target = []
        for cls in classes:
            images = classes[cls]
            flat_images = tuple(map(lambda x: x.flat_data(), images))
            for image in flat_images: 
                training_data[k] = image
                training_target.append(cls)
        training_target = np.array(training_target, dtype=np.int32)

            
        #self.features = training_data[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        lab = Counter()
        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(training_data)):
                for l in self.legalLabels:
                    #print(self.weights[l])
                    lab[l] = self.weights[l].__mul__(training_data[i])
                    #print(lab)
                values = list(lab.values())
                index = values.index(max(values,key= values.count))
                keys = list(lab.keys())
                print(lab)
                print(values)
                print(index)
 				
                if not(training_target[i] == keys[index]):
                    image_counter = Counter()
                    count = 0
                    for column in range(28):
                        for row in range(28):
                            self.weights[training_target[i]][(row,column)] += training_data[i][count]
                            self.weights[keys[index]][(row,column)] -= training_data[i][count]
                            count += 1
                    #print(image_counter)
                    #self.weights[training_target[i]].__radd__(image_counter)
                    #self.weights[keys[index]].__sub__(image_counter)
                    print(self.weights[training_target[i]])                 
                    print(self.weights[training_target[i]][(0,0)])		
                    print(training_target[i])
                    print(keys[index])
                    #self.weights[training_target[i]] += image_counter
                    print(self.weights[training_target[i]])
                    #self.weights[keys[index]] -= image_counter
        return self.validate(validation_data)   					
    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for image, label in data.get_labeled_images():
            #print(image.flat_data())
            vectors = Counter()
            image_counter = Counter()
            flat_img = image.flat_data()
            count = 0
            for column in range(28):
                for row in range(28):
                    #image_counter[(column, row)] = image.flat_data()[count]
                    image_counter[(column, row)] = flat_img[count]
                    count += 1
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * image_counter
                print(vectors[l])
            print("vector", vectors)
            values = list(vectors.values()) 
            keys = list(vectors.keys()) 
            index = values.index(max(values)) 
            guesses.append(keys[index])
            print(keys[index])
        print("here")
        print(guesses)
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []
        featuresWeights = self.weights[label].values()

        return featuresWeights