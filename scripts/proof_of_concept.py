import gym
import main.core.envs as envs
import main.core.portfolio as portfolio
import main.core.asset as asset

SIM_LEN = 1000

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

for i in range(SIM_LEN):
    action = [0.5, 0.5]
    # state, (reward, cost), proceed, _  = env.step(action)
    print(env.step())



