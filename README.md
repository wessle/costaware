### Installation
Set up your preferred environment and activate it, e.g. do
```bash
python -m venv .venv
source .venv/bin/activate
```
After cloning, simply `cd` into the repo and do `pip install -e .`.

### Adding New Dependencies
Add new dependencies by editing the `install_requires` field in setup.py, then do `pip install -e .`.

### TODOs

- [x] Implement `Asset` class 
- [ ] Write unit tests for `Asset` class
- [x] Implement `Portfolio` class
- [ ] Write unit tests for `Portfolio` class
- [ ] Implement OpenAI Gym `env` subclass for financial simulation
- [ ] Write tests for environment
- [ ] Debug `update` method for the Q-learning agent
- [ ] Complete `save/load/checkpoint` methods for Q-learning agent
