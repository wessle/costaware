import argparse
import os
import wesutils
import yaml

import main.core.envs as envs
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict, deque
from datetime import datetime
from defaults import functions, default_configs 
from itertools import product
from shutil import copyfile


class MDPExperiment:

    @staticmethod
    def generate_mdp_env(num_states, num_actions, rewards, costs,
                         transition_seed=None, training_seed=None):
        """
        Construct an arbitrary MDP (by selecting a transition probability matrix
        over the space of all such matrices of given dimension).
        """
        np.random.seed(transition_seed)
    
        states  = list(range(num_states))
        actions = list(range(num_actions))
    
        probs = {}
        for elem in product(states, actions):
            dist = np.random.random(num_states)
            probs[elem] = dist / np.sum(dist)
    
        def transition_matrix(state, action):
            return probs[(state, action)]
    
    
        rewards_fn = functions['rewards'][rewards]
        costs_fn   = functions['costs'][costs]
    
        np.random.seed(training_seed)
    
        env = envs.MDPEnv(states, actions, transition_matrix, rewards_fn, costs_fn)
    
        return states, actions, env

    def __init__(self, env, io, config, **kwargs):
        self.env = env
        self.io = io
        self.config = config
        self.config.update(kwargs)
        self.file_prefix = f's{env.observation_space.n}a{env.action_space.n}'

    def stdout_callback(self, agent, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        print out training information to stdout during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'agent_name': "UnspecifiedAgentName",
            'ratio':      None,
            'step':       None,
        }
        defaults.update(kwargs)
        kwargs = defaults

        print(' '.join([
            f'{kwargs["agent_name"]}',
            f'timestep: {kwargs["step"]:7d}',
            f'(rho={kwargs["ratio"]:.2f}, state={agent.state}, action={agent.action})'
        ]))
    
    def logger_callback(self, agent, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        log training information to a specified log output during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'agent_name': "UnspecifiedAgentName",
            'ratios':     None,  # will break if unchanged, that's good
            'output_dir': self.io.output_dir,
        }
        defaults.update(kwargs)
        kwargs = defaults

        kwargs['filename'] = os.path.join(
            kwargs['output_dir'], 
            f"{self.file_prefix}_{kwargs['agent_name']}_ratios.npy"
        )

        np.save(kwargs['filename'], kwargs['ratios'])
    
    def plot_callback(self, agent, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        plot training information to a specified output directory during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'agent_name': "UnspecifiedAgentName",
            'ratios':     None,  # will break if unchanged, that's good
            'output_dir': self.io.output_dir,
            'xlabel':     'Step',
            'ylabel':     'Ratio'
        }
        defaults.update(kwargs)
        kwargs = defaults

        kwargs['filename'] = os.path.join(
            kwargs['output_dir'], 
            f"{self.file_prefix}_{kwargs['agent_name']}_ratios.png"
        )

        plt.plot(np.arange(len(kwargs['ratios'])), np.array(kwargs['ratios']))
        plt.xlabel(kwargs['xlabel'])
        plt.ylabel(kwargs['ylabel'])
        plt.savefig(kwargs['filename'])
    
    def train(self, agent, **kwargs):
        """
        Train a predefined agent on an initialized environment for a specified
        number of steps. Returns the agent's ratio at each step of the training.
        """
        defaults = {
            'width':           self.config['moving_average_width'],
            'print_interval':  self.config['print_interval'],
            'output_dir':      self.io.output_dir,
            'steps':           self.config['num_steps'],
            'agent_name':      "UnspecifiedAgentName",
            'logger_callback': True,
            'plot_callback':   False,
            'stdout_callback': True,
        }
        defaults.update(kwargs)
        kwargs = defaults

        env.reset()
    
        ratios = []
        rewards = deque(maxlen=kwargs['width'])
        costs = deque(maxlen=kwargs['width'])
    
        for step in range(kwargs['steps']):
            # First, process the agent and the environment
            action = agent.sample_action(env.state)
            next_state, (reward, cost), _, _ = env.step(action)
            agent.update((reward, cost), next_state)

            # Next, process the rewards and costs signals
            rewards.append(reward)
            costs.append(cost)
            ratios.append(np.mean(rewards) / np.mean(costs))
    
            # I/O callbacks
            if step % kwargs['print_interval'] == 0:
                if kwargs['stdout_callback']:
                    self.stdout_callback(
                        agent, step=step, ratio=ratios[-1], 
                        agent_name=kwargs['agent_name']
                    )
    
                if kwargs['logger_callback']:
                    self.logger_callback(
                        agent, step=step, ratios=ratios,
                        output_dir=kwargs['output_dir'],
                        agent_name=kwargs['agent_name']
                    )
    
        # Final I/O callback
        if kwargs['logger_callback']:
            self.logger_callback(
                agent, step=step, ratios=ratios,
                output_dir=kwargs['output_dir'], agent_name=kwargs['agent_name']
            )

        # Plotting callback
        if kwargs['plot_callback']:
            self.plot_callback(agent, ratios=ratios)

        return ratios


class IOManager:

    def __init__(self, config_file):
        self.config_file = config_file
        self.output_config_file = 'config.yml'
        self.output_plot_file = 'plot.py'
        self.input_plot_file = 'scripts/plot.py'

    def load_config(self):
        if self.config_file:
            default_configs.update( wesutils.load_config(self.config_file))
        self.config = default_configs

        return self.config

    def run_startup(self):
        date = datetime.now()

        self.output_dir = os.path.join(args.output, "_".join([
            f"{self.config['env_name']}Comparison",
            f"{date.year:4d}{date.month:02d}{date.day:02d}",
            f"{date.hour:02d}{date.minute:02d}{date.second:02d}"
        ]))

        os.mkdir(self.output_dir)

    def write_config(self):
        keys = [
            'num_states', 'num_actions', 'rewards_fn', 'costs_fn', 'env_name',
            'num_steps', 'moving_average_width', (
                'algorithm_config', [
                    ('LinearAC', ['kwargs']), 
                    ('TabularQ', ['kwargs'])
                ]
            ),
        ]

        # to_write = subdict(self.config, keys)

        filename = os.path.join(self.output_dir, self.output_config_file)
        with open(filename, 'w') as f:
            yaml.dump(to_write, f)

    def copy_plotter(self):
        copyfile(
            self.input_plot_file, 
            os.path.join(self.output_dir, self.output_plot_file)
        )


# argument parsing

parser = argparse.ArgumentParser(
    description='Insert script description here'
)

parser.add_argument(
    '--config', type=str, nargs=1, default="",
    help='YAML filename that provides the configurations for the experiment'
)
parser.add_argument(
    '--output', type=str, nargs=1, default='data',
    help='top-level directory name that all experimental output will be saved'
)

args = parser.parse_args()


if __name__ == "__main__":

    io = IOManager(args.config)

    # Load the configuration file with default configuration handling
    config = io.load_config()
    io.run_startup()
    
    for num_states, num_actions in zip(config['num_states'], config['num_actions']):

        # Create a fixed MDP environment of specified size
        states, actions, env = MDPExperiment.generate_mdp_env(
            num_states, num_actions,
            config["rewards_fn"], config["costs_fn"],
            transition_seed=config["transition_seed"],
            training_seed=config["training_seed"]
        )

        env.reset()
        experiment = MDPExperiment(env, io, config)

        for algorithm in config['algorithm_config']:
            Agent = config['algorithm_config'][algorithm]['class']
            agent = Agent(states, actions, **config['algorithm_config'][algorithm]['kwargs'])

            experiment.train(agent, agent_name=algorithm)

    io.write_config()
    io.copy_plotter()
