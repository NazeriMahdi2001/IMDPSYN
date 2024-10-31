import numpy as np

def floor_decimal(a, precision=0):
    '''
    Floor function, but than with a specific precision
    '''

    return np.round(a - 0.5 * 10 ** (-precision), precision)
