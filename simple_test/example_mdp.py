states = [0, 1]
actions = [0, 1]

probabilities = {
    (0, 0): [1., 0.],
    (0, 1): [0., 1.],
    (1, 0): [1., 0.],
    (1, 1): [0., 0.1]
}

rewards = {
    (s, a): 1 - s for (s, a) in zip(states, actions)
}

costs = {
    (s, a): 1 for (s, a) in zip(states, actions)
}




