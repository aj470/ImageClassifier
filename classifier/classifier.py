

class Classifier:
    def __init__(self, legal_labels):
        self.legalLabels = legal_labels

    @staticmethod
    def name():
        raise NotImplemented()

    def validate(self, validation_data, print_distribution=False):
        guesses = self.classify(validation_data)
        pairs = tuple(zip(guesses, validation_data.get_labels()))
        num_correct = sum(map(lambda x: int(x[0] == x[1]), pairs))
        distribution = [[0] * len(self.legalLabels) for i in range(len(self.legalLabels))]
        for pair in pairs:
            guess, answer = pair
            distribution[answer][guess] += 1
        if print_distribution:
            print("label", end="")
            for i in self.legalLabels:
                print("," + str(i), end="")
            print()
            for i in range(len(distribution)):
                print(i, end=",")
                sep = ""
                for cnt in distribution[i]:
                    print(sep + str(cnt), end="")
                    sep = ","
                print()
        return num_correct / len(validation_data)

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