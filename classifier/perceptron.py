from .classifier import *
from .util import Counter
import random
import numpy as np


""" 
00 01 02 03
10 11 12 13
We know the following:
digit width = 28
digit height = 28
face width = 60
face height = 70
"""

class Perceptron(Classifier):
    def __init__(self, legal_labels):
        self.legalLabels = legal_labels
        self.max_iterations = 10
        self.weights = {}
        digit = True if len(legal_labels) == 10 else False

        for label in legal_labels:
            self.weights[label] = Counter()
            # Initialize Random Weights
            if digit:
                self.height = 28
                self.width = 28
                for row in range(self.height):  # height
                    for column in range(self.width):  # width
                        self.weights[label][(row, column)] = random.uniform(0, 1)
            else:  # face
                self.height = 70
                self.width = 60
                for row in range(self.height):  # height
                    for column in range(self.width):  # width
                        self.weights[label][(row, column)] = random.uniform(0, 1)
                        
    @staticmethod
    def name():
        return "perceptron"

    def _separate(self, dataset):
        classes = {}
        for data, cls in dataset:
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(data)
        return classes

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        classes = self._separate(training_data.get_labeled_images()) # get dictionary with dict['class'] = images 
        training_data = []
        k = 0
        training_target = []
        #separate dict['class'] = images into two lists that is training_data and training_target.
        for cls in classes:
            images = classes[cls]
            flat_images = tuple(map(lambda x: x.flat_data(), images))
            for image in flat_images: 
                training_data.append(image)
                k +=1
                training_target.append(cls)
        training_target = np.array(training_target, dtype=np.int32)

        lab = Counter() #label dictionary
        for iteration in range(self.max_iterations): #for 10 iteration for training.
            print("Starting iteration ", iteration, "...")
            for i in range(len(training_data)):
                #convert image to Counter same as weights.
                image_counter = Counter()                 
                count = 0
                for row in range(self.height):
                    for column in range(self.width):
                        image_counter[(row,column)] = training_data[i][count]
                        count += 1
                #get the prediction      
                for l in self.legalLabels:
                    lab[l] = self.weights[l] * image_counter #get the similarity score.
                #get the best prediction.   
                values = list(lab.values()) 
                keys = list(lab.keys()) 
                index = values.index(max(values))
                # update weights if prediction is not equal to target
                if training_target[i] != keys[index]:
                    self.weights[training_target[i]] += image_counter # add image weight into correct class
                    #self.weights[keys[index]] -= image_counter # subtract image weight from incorrect predicted class
                    
        return self.validate(validation_data)   					
    def classify(self, data ):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        guesses = []
        for image, label in data.get_labeled_images():
            vectors = Counter() #for the label predictions
            image_counter = Counter() # for converting image to counter.
            flat_img = image.flat_data()
            count = 0
            for row in range(self.height):
                for column in range(self.width):
                    image_counter[(row, column)] = flat_img[count]
                    count += 1

            for l in self.legalLabels:
                vectors[l] = self.weights[l] * image_counter #get the similarity score.
            #get the best prediction
            values = list(vectors.values()) 
            keys = list(vectors.keys()) 
            index = values.index(max(values)) 
            guesses.append(keys[index])
        return guesses