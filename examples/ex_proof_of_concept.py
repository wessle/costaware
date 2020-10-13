from main.core import asset, envs, portfolio

print("\n".join([
    "This example demonstrates as a ~~proof of concept~~ that all of the",
    "custom classes and environments are interacting together as designed",
    "along the most high level basic \"happy paths\".",
    "",
    "This is ~~not~~ a substitute for proper unit tests, David!"
]))

risky_asset = asset.Asset(10., 0.50, 0.0)
riskless_asset = asset.Asset(10., 0.25, 0.00)

portfolio = portfolio.Portfolio([risky_asset, riskless_asset], [0.5, 0.5], 100)

env = envs.SharpeCostAwareEnv(portfolio)
env.reset()

for i in range(100):
    action = env.portfolio.weights
    state, (reward, cost), terminate, _ = env.step(action)
    print(f"Step {i}... {reward:0.3f} {cost:0.3f}")

