#!/usr/bin/env python3

import numpy, scipy

if __name__ == '__main__':
    a = scipy.empty((5,3,7), dtype=int)
    b = scipy.arange(5).reshape((5,1))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a[i,j,k] = 1000 * i + 100 * j + 10 * k

    print('a: ' + repr(a) + ', shape = ' + repr(a.shape))
    print('a.T: ' + repr(a.T) + ', shape = ' + repr(a.T.shape))
    print('b: ' + repr(b))
    print('(a.T + b.T).T: ' + repr((a.T + b.T).T))
    print('b + b: ' + repr(b + b))
