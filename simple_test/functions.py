import numpy as np


def r1(s,a):
    return s**2

def c1(s,a):
    return max(1, a**2)


def r2(s,a):
    return s*a

def c2(s,a):
    return 1 / max(1, s*a)


def r3(s, a):
    return (s % 2) * (a % 2)

def c3(s, a):
    return 1 + (s - 1)**2 + (a - 1)**2
