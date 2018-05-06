from .classifier import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class CustomClassifier(Classifier):
    def __init__(self, legal_labels):
        """KNN Classifier"""
        self.legal_labels = legal_labels
        self.legal_labels_len = len(legal_labels)
        
    @staticmethod
    def name():
        return "custom - KNN"
        
    def _separate(self, dataset):
        classes = {}
        for data, cls in dataset:
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(data)
        return classes

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        #print(validation_data.get_labeled_images())
        #training_dataset = {}
        training_examples = 0
        training_dataset = []
        training_target = []
        classes = self._separate(training_data.get_labeled_images())
        for cls in classes:
            images = classes[cls]               # images assigned to each class
            flat_images = tuple(map(lambda x: x.flat_data(), images))  # flat images
            for img in flat_images:
                training_examples += 1
                training_dataset.append(img)
                training_target.append(cls)
        training_dataset = np.array(training_dataset, dtype=np.int32)
        training_target = np.array(training_target, dtype=np.int32)
        
        #print(training_examples)
        #print(len(training_dataset),len(training_target))
        #print(training_dataset[0])        
        #print(self.legal_labels)
            
        
        validation_dataset = []
        validation_target = []
        validation_examples = 0
        classes = self._separate(validation_data.get_labeled_images())
        for cls in classes:
            images = classes[cls]               # images assigned to each class
            flat_images = tuple(map(lambda x: x.flat_data(), images))  # flat images
            for img in flat_images:
                validation_examples += 1
                validation_dataset.append(img)
                validation_target.append(cls)
        validation_dataset = np.array(validation_dataset, dtype=np.int32)
        validation_target = np.array(validation_target, dtype=np.int32)
        
        
        
        
        euclidean_distance_validation = np.zeros((validation_examples, training_examples), dtype=np.int32)
        sorted_distance_class = np.zeros((validation_examples, training_examples), dtype=np.int32)
        
        # Calculates eculidian data for testing data
        for i in range(0, validation_examples):
            euclidean_distance_validation[i:i + 1, :] = np.sqrt(np.sum(np.square(training_dataset[:, :] - validation_dataset[i, :]), axis=1))
        
        
        for j in range(0,validation_examples):
           sorted_distance_class[j] = mylist = [training_target[i] for i in np.argsort(euclidean_distance_validation[j])]
           #training_target[:][np.argsort(euclidean_distance_validation[j])]
        
        np.savetxt("label.csv", sorted_distance_class, delimiter=",")
        np.savetxt("target.csv", validation_target, delimiter=",")
        
        k = [1,2,3,4,5,6,7,8,9,10]
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
            

        # plotting the graphs on same plane for errors vs value of K
        plt.xlabel('Value of K')
        plt.ylabel('Error')
        # red represents the test error
        # green represents the traning error
        red_box = mpatches.Patch(color='red', label='Test Error')
        green_box = mpatches.Patch(color='green', label='Traning Error')

        plt.legend(handles=[red_box,green_box])

        plt.plot(k, testing_error[0], color='red', marker='*')
        plt.title('knn Classifier')
        plt.show()
       

        return 

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        #raise NotImplemented()