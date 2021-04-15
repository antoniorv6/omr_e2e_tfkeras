from data_load import load_fold
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program arguments to work")
    parser.add_argument('--corpus', type=str, help="Corpus to be processed")
    parser.add_argument('--codif', type=str, help="Codification to use")

    args = parser.parse_args()
    return args

arguments = parse_arguments()

runningwords = 0

for i in range(0,10):
    X, Y = load_fold(arguments.corpus, i, arguments.codif)
    runningwords += np.sum([len(sequence) for sequence in Y])

print(runningwords)  