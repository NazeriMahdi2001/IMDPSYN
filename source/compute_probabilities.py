#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
from source.commons import floor_decimal
from source.tabulate_scenario import create_table

def computeRegionIdx(points, lb, ub, regions_per_dimension, size_per_region, ubBorderInside=True):
    '''
    Function to compute the indices of the regions that a list of points belong

    Parameters
    ----------
    points : 2D Numpy array (m x n)
        Array, with every row being a point to determine the center point for.
    lb, ub: 1D Numpy array (n)
        Lower left corner and upper right corner of state space partition.
    number_per_dimension: 1D Numpy array (n)
        Number of regions/elements per dimension of the state space.
    regions_per_dimension: 1D Numpy array (n)
        Size/width of each region/element per dimension of the state space.
    ubBorderInside: Boolean
        If True, then count a sample that is exactly on the upper boundary of a region to be contained in it

    Returns
    -------
    2D Numpy array
        Array, with every row being the indices.

    '''

    # Shift the points such that the smallest point within the partition is zero.
    pointsZero = points - lb

    # Compute indices to which each point belongs
    indices = (pointsZero // size_per_region).astype(int)

    # Reduce index by one if it is exactly on the border
    indices -= ((pointsZero % size_per_region == 0).T * ubBorderInside).T

    indices_nonneg = np.minimum(np.maximum(0, indices), np.array(regions_per_dimension) - 1).astype(int)

    return indices, indices_nonneg


def compute_intervals(etc, Nsamples, inverse_confidence, partition, clusters, probability_table):
    '''
    Compute the probability intervals P(s,a,s') for a given pair (s,a) and for all successor states s'.
    '''
    debug=0
    # Initialize arrays to keep track of how many samples are contained in each partition element
    counts_lb = np.zeros(partition['regions_per_dimension'])
    counts_ub = np.zeros(partition['regions_per_dimension'])

    # For each cluster of samples, determine between which indices (dimension-wise) it is contained
    # Note: iMin (iMax) is the same as imin (imax), but than with indices restricted to within the partition
    imin, iMin = computeRegionIdx(clusters['lb'], partition['lb'], partition['ub'], partition['regions_per_dimension'],
                                 partition['size_per_region'])
    imax, iMax = computeRegionIdx(clusters['ub'], partition['lb'], partition['ub'], partition['regions_per_dimension'],
                                 partition['size_per_region'])

    if debug:
        print('imin:', imin)
        print('imax:', imax)

    # Epsilon for Hoeffding's inequality
    epsilon = np.sqrt(1 / (2 * Nsamples) * np.log(
        2 / inverse_confidence))

    counts_goal_lb = 0
    counts_goal_ub = 0

    counts_unsafe_lb = 0
    counts_unsafe_ub = 0

    ###

    # Compute number of samples fully outside of partition.
    # If all indices are outside the partition, then it is certain that
    # this sample is outside the partition
    fully_out = (imax < 0).any(axis=1) + (imin > partition['regions_per_dimension'] - 1).any(axis=1)
    if debug:
        print('Number of samples fully outside of partition:', fully_out.sum())

    counts_outOfPartition_lb = clusters['value'][fully_out].sum().astype(int)

    # Determine samples that are partially outside
    partially_out = (imin < 0).any(axis=1) + (imax > partition['regions_per_dimension'] - 1).any(axis=1)
    if debug:
        print('Number of samples partially outside of partition:', partially_out.sum())

    counts_outOfPartition_ub = clusters['value'][partially_out].sum().astype(int)

    # If we precisely know where the sample is, and it is not outside the
    # partition, then add one to both lower and upper bound count
    in_single_region = (imin == imax).all(axis=1) * np.bitwise_not(fully_out)
    if debug:
        print('Number of samples within a single region:', in_single_region.sum())
        print('\nAnalyse samples within a single region...')
    for key, val in zip(imin[in_single_region],
                        clusters['value'][in_single_region]):
        key = tuple(key)
        if debug:
            print('- Index',key)
        if key in partition['goal_idx']:
            if debug:
                print('-- Is a goal state')
            counts_goal_lb += val
            counts_goal_ub += val

        elif key in partition['unsafe_idx']:
            if debug:
                print('-- Is an unsafe state')
            counts_unsafe_lb += val
            counts_unsafe_ub += val

        else:
            counts_lb[key] += val
            counts_ub[key] += val

    # Remove the samples we have analysed
    keep = ~in_single_region & ~fully_out
    iMin = iMin[keep]
    iMax = iMax[keep]
    c_rem = np.arange(len(clusters['value']))[keep]

    ###

    # For the remaining samples, only increment the upper bound count
    # print(f'\nAnalyse remaining {len(c_rem)} samples...')
    for x, c in enumerate(c_rem):
        intersects_with = tuple(map(slice, iMin[x], iMax[x] + 1))
        # print(f'- Sample {c} with iMin={iMin[x]} and iMax={iMax[x]} intersects with region slice: {intersects_with}')
        counts_ub[intersects_with] += clusters['value'][c]

        index_tuples = set(itertools.product(*map(range, iMin[x], iMax[x] + 1)))

        # Check if all are goal states
        if index_tuples.issubset(partition['goal_idx']) and not partially_out[c]:
            # print('-- All are goal states')
            counts_goal_lb += clusters['value'][c]
            counts_goal_ub += clusters['value'][c]

        # Check if all are unsafe states
        elif index_tuples.issubset(partition['unsafe_idx']) and not partially_out[c]:
            # print('-- All are unsafe states')
            counts_unsafe_lb += clusters['value'][c]
            counts_unsafe_ub += clusters['value'][c]

        # Otherwise, check if part of them are goal/unsafe states
        else:
            if not index_tuples.isdisjoint(partition['goal_idx']):
                # print('-- Some are goal states')
                counts_goal_ub += clusters['value'][c]

            if not index_tuples.isdisjoint(partition['unsafe_idx']):
                # print('-- Some are unsafe states')
                counts_unsafe_ub += clusters['value'][c]

    ###

    # Extract the counts of all non-goal/unsafe states that contain one or more samples
    counts_nonzero = [[partition['tup2idx'][tup], counts_lb[tup], count_ub]
                      for tup, count_ub in np.ndenumerate(counts_ub) if count_ub > 0
                      and tup not in partition['goal_idx']
                      and tup not in partition['unsafe_idx']]

    # Add the counts for the unsafe and goal state (note: counts for these states are merged into a single state)
    counts_header = []
    if len(partition['unsafe_idx']) > 0:
        counts_header += [[-2, counts_unsafe_lb, counts_unsafe_ub]] # IMDP state -2 is unsafe state
    if len(partition['goal_idx']) > 0:
        counts_header += [[-1, counts_goal_lb, counts_goal_ub]] # IMDP state -1 is the goal state

    counts = np.array(counts_header + counts_nonzero, dtype=int)

    # Compute probability intervals for transitioning outside the partition
    outOfPartition_lb = np.maximum(0, counts_outOfPartition_lb / Nsamples - epsilon)
    outOfPartition_ub = 1 - probability_table[counts_outOfPartition_ub, 0] # TODO: Integrate with this table

    # Counts can have length zero if all samples are outside the partition
    if len(counts) > 0:
        # UB nr. of discarded samples (as per scenario approach) is total nr. of samples minus the LB nr. we counted
        discarded_samples_ub = np.minimum(Nsamples - counts[:, 1], Nsamples)

        # Compute lower bound probability
        probability_lb = probability_table[discarded_samples_ub, 0] # TODO: Integrate with this table

        # Compute upper bound probability either with Hoeffding's bound
        # TODO: Check how we approach that here (scenario approach vs. Hoeffding's)
        probability_ub = np.minimum(1, counts[:, 2] / Nsamples + epsilon)

        # Take average sample count to compute approximate (precise) transition probabilities
        probability_approx = counts[:, 1:3].mean(axis=1) / Nsamples
        successor_idxs = counts[:, 0]

    else:
        probability_lb = np.array([])
        probability_ub = np.array([])
        probability_approx = np.array([])
        successor_idxs = np.array([])

    nr_decimals = 5 # Number of decimals to use for probability intervals (more decimals means larger file sizes)
    Pmin = 1e-8 # Minimum lower bound probability interval to use (PRISM cannot handle zero lower bounds)

    #### PROBABILITY INTERVALS
    probs_lb = np.maximum(Pmin, floor_decimal(probability_lb, nr_decimals))
    probs_ub = np.minimum(1, floor_decimal(probability_ub, nr_decimals))

    # Create interval strings (only entries for prob > 0)
    interval_strings = ["[" +
                        str(lb) + "," +
                        str(ub) + "]"
                        for (lb, ub) in zip(probs_lb, probs_ub)]

    # Compute probability interval to go outside the partition
    outOfPartition_lb = np.maximum(Pmin, floor_decimal(outOfPartition_lb, nr_decimals))
    outOfPartition_ub = np.minimum(1, floor_decimal(outOfPartition_ub, nr_decimals))
    outOfPartition_string = '[' + \
                      str(outOfPartition_lb) + ',' + \
                      str(outOfPartition_ub) + ']'

    # #### POINT ESTIMATE PROBABILITIES
    # probability_approx = np.round(probability_approx, nr_decimals)

    # # Create approximate prob. strings (only entries for prob > 0)
    # approx_strings = [str(p) for p in probability_approx]

    # # Compute approximate transition probability to go outside the partition
    # outOfPartition_approx = np.round(1 - sum(probability_approx), nr_decimals)

    returnDict = {
        'successor_idxs': successor_idxs,
        'interval_strings': interval_strings,
        # 'approx_strings': approx_strings,
        'outOfPartition_interval_string': outOfPartition_string,
        # 'outOfPartition_approx': outOfPartition_approx,
    }

    return [etc, returnDict]
