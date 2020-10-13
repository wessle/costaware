import numpy as np


# Functions for use in defining synthetic cost-aware MDPs

goal_state = 0


def r1(s, a):
    return s**3

def r2(s, a):
    return s + a

def r3(s, a):
    return (s % 2) * (a % 2)

def r4(s, a):
    return 100 * (s == goal_state)

def r5(s, a):
    return 1 * (s % 2 == 0)


def c1(s, a):
    return max(1, s * a)

def c2(s, a):
    return 1 / max(1, s*a)

def c3(s, a):
    return 1 + (a % 3 - 1)**2

def c4(s, a):
    return np.exp(-s)

def c5(s, a):
    return 1

def c6(s, a):
    return 1 * (s == goal_state) + 10 * (s != goal_state)
