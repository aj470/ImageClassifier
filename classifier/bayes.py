from .classifier import *


class BayesClassifier(Classifier):
    def __init__(self, legal_labels):
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

            for i in range(dimension):
                for j in range(len(flat_images)):
                    if flat_images[j][i] > 0:
                        sums[i] += 1
                    else:
                        sums[i] += 0

            for i in range(dimension):
                sums[i] = (sums[i] + alpha) / (len(images) + alpha)

            self.class_parameters[cls] = sums
            self.priors[cls] = len(images) / total_size

        return self.validate(validation_data)

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        guesses = []
        for image, label in data.get_labeled_images():
            vector = image.flat_data()
            probabilities = {}

            for cls, params in self.class_parameters.items():
                probabilities[cls] = 1
                for i in range(len(vector)):
                    if vector[i] > 0:
                        probabilities[cls] *= params[i]
                    else:
                        probabilities[cls] *= 1 - params[i]

            guesses.append(max(probabilities.items(), key=lambda x: x[1])[0])
        return guesses