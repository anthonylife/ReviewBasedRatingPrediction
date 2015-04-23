#!/usr/bin/env python
#encoding=utf8

import numpy as np

def test(a):
    a= a/np.sum(a)
    print a

if __name__ == "__main__":
    b = np.array([1,2,3])
    test(b)
    print b

