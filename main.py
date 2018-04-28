#!/user/bin/python3

import argparse
import datasets
import sys

from time import monotonic
from classifier import *

# dimensions in characters
# widths aren't used
FACE_WIDTH = 60
FACE_HEIGHT = 70

DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28


def face_datasets():
    return [
        datasets.load_dataset("facedata/facedatatrain", "facedata/facedatatrainlabels", FACE_HEIGHT),
        datasets.load_dataset("facedata/facedatavalidation", "facedata/facedatavalidationlabels", FACE_HEIGHT),
        datasets.load_dataset("facedata/facedatatest", "facedata/facedatatestlabels", FACE_HEIGHT),
        "faces"
    ]


def digit_datasets():
    return [
        datasets.load_dataset("digitdata/trainingimages", "digitdata/traininglabels", DIGIT_HEIGHT),
        datasets.load_dataset("digitdata/validationimages", "digitdata/validationlabels", DIGIT_HEIGHT),
        datasets.load_dataset("digitdata/testimages", "digitdata/testlabels", DIGIT_HEIGHT),
        "digits"
    ]


def run_test(datasets, algorithm):
    training, validation, test, dataname = datasets
    percentages = map(lambda x: x/10, range(1, 11))

    datapoints = []
    for percent in percentages:
        count = int(percent * len(training))
        training_set = training.subset(count)
        start_time = monotonic()
        print(count)
        algorithm.train(training_set, validation)
        end_time = monotonic()

        accuracy = algorithm.validate(test)
        datapoints.append((dataname, count, end_time - start_time, accuracy))

    return datapoints


def main():
    parser = argparse.ArgumentParser(description="CS440 Project 2 -- Image Classification")
    parser.add_argument("-f", "--full-eval", help="Run all algorithms on both faces and digits",
                        action='store_true', default=None)
    parser.add_argument("-a", "--algorithm", type=int, choices=[1, 2, 3, 4, 5],
                        help="1. Perceptron, 2. Naive Bayes Classifier, 3. Custom", default=None)
    parser.add_argument("-d", "--data", type=str, choices=["face", "digit"],
                        help="whether to evaluate faces or digits", default=None)
    parser.add_argument("-c", "--csv", help="Output CSV data",
                        action='store_true', default=None)
    args = vars(parser.parse_args())

    algorithms = [Perceptron, BayesClassifier, CustomClassifier, BasicClassifier, PerfectClassifier]

    full_eval = args["full_eval"]
    algorithm = args["algorithm"]
    datasource = args["data"]
    csv = args["csv"]

    datapoints = {}

    # evaluate run options
    if full_eval and (algorithm or datasource):
        # too many options specified
        print("You cannot specify -a or -d with the -f option.\nExiting.")
        return

    elif full_eval:
        # full evaluation
        faces = face_datasets()
        digits = digit_datasets()
        # run all algorithms on both faces and digits
        for alg in algorithms:
            data = run_test(faces, alg([0, 1]))
            data.extend(run_test(digits, alg(list(range(0, 10)))))
            # record data points for this algorithm
            datapoints[alg.name()] = data

    elif not (algorithm or datasource):
        # no arguments
        print("No arguments provided.")
        print("Example usage: ./main.py -a 1 -d face\n")
        parser.print_help(sys.stdout)
        print("Exiting.")
        return

    elif not (algorithm and datasource):
        # didn't specify the algorithm or what data to execute on
        print("Must specify both -a and -d.\nExiting.")

    else:
        # both algorithm and data set is provided
        # get correct dataset and labels
        data = {"face": face_datasets, "digit": digit_datasets}[datasource]()
        labels = {"face": [0, 1], "digit": list(range(0, 10))}[datasource]
        # get specific algorithm
        algorithm = algorithms[algorithm-1](labels)
        # evaluate algorithm on the given data
        datapoints[algorithm.name()] = run_test(data, algorithm)

    # output results
    if csv:
        print("algorithm,data type,training set size,training time,accuracy")
        for algorithm in datapoints.keys():
            if len(datapoints[algorithm]) == 0:
                continue

            for entry in datapoints[algorithm]:
                print(algorithm, entry[0], entry[1], entry[2], entry[3], sep=",")
    else:
        for algorithm in datapoints.keys():
            if len(datapoints[algorithm]) == 0:
                continue

            for entry in datapoints[algorithm]:
                print("Algorithm:     ", algorithm)
                print("Data type:     ", entry[0])
                print("Training size: ", entry[1])
                print("Training time: ", entry[2])
                print("Accuracy:      ", entry[3], "\n")


if __name__ == "__main__":
    main()
