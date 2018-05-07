from .classifier import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class CustomClassifier(Classifier):
    def __init__(self, legal_labels):
        """KNN Classifier"""
        self.legalLabels = legal_labels
        self.legal_labels_len = len(self.legalLabels)
        self.training_dataset = []
        self.training_target = []
        self.training_examples = 0
        self.k = 0
        
    @staticmethod
    def name():
        return "Custom - KNN"
        
    def _separate(self, dataset):
        classes = {}
        for data, cls in dataset:
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(data)
        return classes

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        
        training_examples = 0
        training_dataset = []
        training_target = []
        classes = self._separate(training_data.get_labeled_images())
        for cls in classes:
            images = classes[cls]               
            flat_images = tuple(map(lambda x: x.flat_data(), images))
            for img in flat_images:
                training_examples += 1
                training_dataset.append(img)
                training_target.append(cls)
        self.training_dataset = training_dataset
        self.training_target = training_target
        self.training_examples = training_examples
        training_dataset = np.array(training_dataset, dtype=np.int32)
        training_target = np.array(training_target, dtype=np.int32)
        
            
        
        validation_dataset = []
        validation_target = []
        validation_examples = 0
        classes = self._separate(validation_data.get_labeled_images())
        for cls in classes:
            images = classes[cls]               
            flat_images = tuple(map(lambda x: x.flat_data(), images))  
            for img in flat_images:
                validation_examples += 1
                validation_dataset.append(img)
                validation_target.append(cls)
        validation_dataset = np.array(validation_dataset, dtype=np.int32)
        validation_target = np.array(validation_target, dtype=np.int32)
        
        
        
        
        euclidean_distance_validation = np.zeros((validation_examples, training_examples), dtype=np.int32)
        sorted_distance_class = np.zeros((validation_examples, training_examples), dtype=np.int32)
        
        
        for i in range(0, validation_examples):
            euclidean_distance_validation[i:i + 1, :] = np.sqrt(np.sum(np.square(training_dataset[:, :] - validation_dataset[i, :]), axis=1))
        
        
        for j in range(0,validation_examples):
           sorted_distance_class[j] = [training_target[i] for i in np.argsort(euclidean_distance_validation[j])]
           
        
   
        
        if len(self.legalLabels) > 2:
            k = [1,2,3,4,5,6,7,8,9,10]
        else:
            k = [x for x in range(int(training_examples/4),int(training_examples/2)) if x%20 == 0]
        print("K list", k)
        testing_error = np.zeros((1, len(k)), dtype=np.float)
        predicted_label = [] 
        for item in k:
            error = 0
            for i in range(0, validation_examples):
                lst = list(sorted_distance_class[i][0:item])
                predicted_label.append(max(lst,key= lst.count))
                if predicted_label[i] != validation_target[i]:
                    error += 1
            testing_error[0, k.index(item)] = error/validation_examples
            
        self.k = k[list(testing_error[0]).index(min(testing_error[0]))]
        print(self.k)
        
        plt.xlabel('Value of K')
        plt.ylabel('Error')
        
        red_box = mpatches.Patch(color='red', label='Test Error')
        green_box = mpatches.Patch(color='green', label='Traning Error')

        plt.legend(handles=[red_box,green_box])

        plt.plot(k, testing_error[0], color='red', marker='*')
        plt.title('knn Classifier')
        plt.savefig(str(training_examples)+'.pdf')

        return self.validate(validation_data)

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        training_dataset = np.array(self.training_dataset, dtype=np.int32)
        training_target = np.array(self.training_target, dtype=np.int32)
        training_examples = self.training_examples
        classify_dataset = []
        guesses = []
        classify_count = 0
        flat_images = tuple(map(lambda x: x.flat_data(), data.get_images()))  # flat images
        for img in flat_images:
            classify_count += 1
            classify_dataset.append(img)
        classify_dataset = np.array(classify_dataset, dtype=np.int32)
        
        euclidean_distance_validation = np.zeros((classify_count, training_examples), dtype=np.int32)
        sorted_distance_class = np.zeros((classify_count, training_examples), dtype=np.int32)
        
        
        for i in range(0, classify_count):
            euclidean_distance_validation[i:i + 1, :] = np.sqrt(np.sum(np.square(training_dataset[:, :] - classify_dataset[i, :]), axis=1))
        
        
        for j in range(0,classify_count):
           sorted_distance_class[j] = [training_target[i] for i in np.argsort(euclidean_distance_validation[j])]
           
        
        k = self.k
        print("K is", k)
        for i in range(0, classify_count):
            lst = list(sorted_distance_class[i][0:k])
            guesses.append(max(lst,key= lst.count))
        return guesses