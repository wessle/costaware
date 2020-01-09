import gym
import main.core.agents as agent
import main.core.asset as asset
import main.core.envs as envs
import main.core.portfolio as portfolio
import main.utils.simplex as simplex
import numpy as np
import torch

from collections import OrderedDict

def unpack(item):
    def optional_dict(iterable):
        if isinstance(iterable, (dict, OrderedDict)):
            return iterable.values()
        else:
            return iterable

    def kernel(iterable):
        it = iter(optional_dict(iterable))
        for e in it:
            if isinstance(e, (list, tuple, dict, OrderedDict, np.ndarray)):
                for f in kernel(e):
                    yield f
            else:
                yield e

    return list(kernel(item))


SIM_LEN = 100

env = envs.SortinoCostAwareEnv(
    portfolio.Portfolio(
        [
            asset.Asset(1., 0.05, 0.),
            asset.Asset(1., 0.025, 0.)
        ], 
        [0.5, 0.5], 
        1000.
    ),
    0.
)
env.reset()

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        ).to(device)

agent = agent.RVIQLearningBasedAgent(
    1000000,                          # buffer_maxlen
    256,                              # batchsize
    simplex.generate_simplex(2, 10),  # actions
    model,                            # q_network
    1e-3,                             # q_lr
    1e-3,                             # rho_lr
)

for i in range(SIM_LEN):
    unpacked_state = unpack(env.state)
    action = agent.sample_action(unpacked_state)
    state, reward_cost_tuple, proceed, _  = env.step(action)
    agent.update(reward_cost_tuple, state)

