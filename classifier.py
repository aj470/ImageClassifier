from sklearn.naive_bayes import BernoulliNB as nb
import numpy as np


class Classifier:
    def __init__(self, legal_labels):
        self.legal_labels = legal_labels

    @staticmethod
    def name():
        raise NotImplemented()

    def validate(self, validation_data):
        guesses = self.classify(validation_data)
        pairs = zip(guesses, validation_data.get_labels())
        num_correct = sum(map(lambda x: int(x[0] == x[1]), pairs))
        return num_correct / len(validation_data)

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        raise NotImplemented()

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        raise NotImplemented()


class Perceptron(Classifier):
    @staticmethod
    def name():
        return "perceptron"

    def train(self, training_data, validation_data):
        """Must return the final percentage accuracy achieved on validation data set."""
        raise NotImplemented()

    def classify(self, data):
        """Must return an array of predictions, each prediction corresponding to the image at that index."""
        raise NotImplemented()


class BayesClassifier(Classifier):
    def __init__(self, legal_labels):
        super().__init__(legal_labels)
        self.model = nb()

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


class BasicClassifier(Classifier):
    def __init__(self, legal_labels):
        super().__init__(legal_labels)
        self.guess = None

    @staticmethod
    def name():
        return "basic"

    def train(self, training_data, validation_data):
        labels = {}
        for label in training_data.get_labels():
            # count occurrences of each label
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1

        # pick the most frequent label as the guess for all images
        self.guess = max(labels.items(), key=lambda x: x[1])[0]
        return self.validate(validation_data)

    def classify(self, data):
        return [self.guess for elem in data]


class PerfectClassifier(Classifier):
    @staticmethod
    def name():
        return "perfect"

    def train(self, training_data, validation_data):
        return self.validate(validation_data)

    def classify(self, data):
        return [elem[1] for elem in data]