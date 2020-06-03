#!/usr/bin/env python

import pickle
import pandas as pd
import argparse
from os import path
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Directory to import to, usually 'data/'")
    args = parser.parse_args()
    datapath = args.location
    tusers_evalpath = path.join(datapath, "tusers_eval.p") 
    tusers_eval = pickle.load(open(tusers_evalpath, "rb"))
    
    #ignore negative outliers
    tusers_eval_filtered = {}
    for key, value in tusers_eval.items():
        if value >= 0:
            tusers_eval_filtered[key] = value
    
    
    scores = pd.Series(tusers_eval_filtered)

    scores.plot.hist(grid=True, bins=30, rwidth=0.9, color='#607c8e')
    plt.title('Evaluation score histogram')
    plt.xlabel('Counts')
    plt.ylabel('Score')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('scores-hist.png', bbox_inches='tight')