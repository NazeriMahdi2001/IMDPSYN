#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import binomtest
import itertools
from source.commons import floor_decimal

def compute_intervals(etc, roi, absDimension, Nsamples, inverse_confidence, partition, clusters):
    '''
    Compute the probability intervals P(s,a,s') for a given pair (s,a) and for all successor states s'.
    '''
    debug=0
    # Initialize arrays to keep track of how many samples are contained in each partition element
    counts_lb = np.zeros(partition['nr_regions'] + 2, dtype=int)
    counts_ub = np.zeros(partition['nr_regions'] + 2, dtype=int)

    nr_decimals = 5 # Number of decimals to use for probability intervals (more decimals means larger file sizes)
    Pmin = 1e-12 # Minimum lower bound probability interval to use (PRISM cannot handle zero lower bounds)

    for lb, ub in zip(clusters['lb'], clusters['ub']):
        tuples = set(itertools.product(*map(range, lb, ub + 1)))
    
        intersections = set()
        for index in tuples:
            if np.all(np.array(index) >= 0) and np.all(np.array(index) < absDimension):
                if index in partition['goal_idx']:
                    intersections.add(-1)
                elif index in partition['unsafe_idx'] or partition['tup2idx'][index] not in roi:
                    intersections.add(-2)
                else:
                    intersections.add(partition['tup2idx'][index])
            else:
                intersections.add(-2)

        if len(intersections) == 1:
            counts_lb[list(intersections)[0]] += 1
            counts_ub[list(intersections)[0]] += 1
        elif len(intersections) > 1:
            for ind in intersections:
                counts_ub[ind] += 1


    successor_idxs = []
    probs_lb = []
    probs_ub = []

    for i in roi:
        binom = binomtest(k=counts_lb[i], n=Nsamples)
        probs = binom.proportion_ci(confidence_level = 1 - inverse_confidence/Nsamples)
        low =   probs.low
        
        binom = binomtest(k=counts_ub[i], n=Nsamples)
        probs = binom.proportion_ci(confidence_level = 1 - inverse_confidence/Nsamples)
        high =  probs.high

        successor_idxs.append(i)
        probs_lb.append(np.maximum(Pmin, floor_decimal(low, nr_decimals)))
        probs_ub.append(np.minimum(1, floor_decimal(high, nr_decimals)))
            

    # Create interval strings (only entries for prob > 0)
    interval_strings = ["[" +
                        str(lb) + "," +
                        str(ub) + "]"
                        for (lb, ub) in zip(probs_lb, probs_ub)]

    returnDict = {
        'successor_idxs': successor_idxs,
        'interval_strings': interval_strings,
    }

    return [etc, returnDict]
