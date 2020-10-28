import os 
import yaml

import numpy as np
import main.core.envs as envs
import main.utils.defaults as defaults
import matplotlib.pyplot as plt

from collections import deque
from itertools import product


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
    
        if isinstance(rewards, str):
            rewards = defaults.__dict__[rewards]

        if isinstance(costs, str):
            costs = defaults.__dict__[costs]
    
        np.random.seed(training_seed)
    
        env = envs.MDPEnv(states, actions, transition_matrix, rewards, costs)
    
        return states, actions, env


class IOManager:
    """
    """

    def __init__(self, output_dir):
        os.mkdir(output_dir)
        self.output_dir = output_dir

    def to_stdout(self, msg):
        print(msg)

    def save_yml(self, filename, dictionary):
        with open(os.path.join(self.output_dir, filename), 'w') as out:
            out.write(yaml.safe_dump(dictionary))

    def save_npy(self, filename, array):
        np.save(os.path.join(self.output_dir, filename), array)

    def save_plt(self, filename, fig, **kwargs):
        fig.savefig(os.path.join(self.output_dir, filename), **kwargs) 


class TrialRunner:
    """
    Given an environment and an agent, do training and store the results.
    """

    def __init__(self, env, agent, io_manager, **kwargs):
        """
        Params
        ------
        env        : an OpenAI gym environment (instantiated)
        agent      : an agent (instantiated)
        io_manager : an IOManager (instantiated)
        kwargs     : dict 
         - width (default = 100)
           The width of the simple moving average window for computation of the
           reward+cost ratios.
         - io_interval (default = 10000)
           How frequently the agent should print provisional data to stdout and
           log to save files.
         - n_steps (default = 500000)
           Number of training steps.
         - logging (default = True)
           Whether the experiment will save ratios to a file.
         - plotting (default = False)
           Whether the experiment will save plot summaries of the ratios.
         - stdouting (default = True)
           Uhether the experiment will print provisional info to stdout.
        """
        self.env   = env
        self.agent = agent
        self.io    = io_manager

        defaults = {
            'width':          100,
            'print_interval': 10_000,
            'n_steps':        500_000,
            'logging':        True,
            'plotting':       False,
            'stdouting':      True,
            'agent_name':     type(agent).__name__,
        }
        defaults.update(kwargs)
        self.update(**defaults)

    def update(self, **kwargs):
        """
        Sends every specified keyword argument to an attribute of the object.
        That is, if foo=bar is passed into the kwargs, then after running this
        method, self.foo == bar
        """
        self.__dict__.update(kwargs)

    def stdout_callback(self, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        print out training information to stdout during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'ratio': None,
            'step':  None,
        }
        defaults.update(kwargs)

        output_message = ' '.join([
            f'{self.agent_name}',
            f'timestep: {kwargs["step"]:7d}',
            f'(rho={kwargs["ratio"]:.2f}, state={self.agent.state}, action={self.agent.action})'
        ])

        self.io.to_stdout(output_message)
    
    def logger_callback(self, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        log training information to a specified log output during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'ratios':     None,  # will break if unchanged, that's good
        }
        defaults.update(kwargs)

        filename = f"{self.agent_name}_ratios.npy"

        self.io.save_npy(filename, kwargs['ratios'])
    
    def plot_callback(self, **kwargs):
        """
        This function is called at each step of the training loop to (selectively)
        plot training information to a specified output directory during the loop.
    
        Any callback with this signature may be used instead, but this is a
        reasonable default behavior.
        """
        defaults = {
            'ratios':     None,  # will break if unchanged, that's good
            'xlabel':     'Step',
            'ylabel':     'Ratio'
        }
        defaults.update(kwargs)

        filename = f"{self.agent_name}_ratios.png"

        fig, ax = plt.subplots()

        ax.plot(np.arange(len(kwargs['ratios'])), np.array(kwargs['ratios']))
        ax.set_xlabel(kwargs['xlabel'])
        ax.set_ylabel(kwargs['ylabel'])

        self.io.save_plt(filename, fig)
    
    def train(self):
        """
        Train a predefined agent on an initialized environment for a specified
        number of steps. Returns the agent's ratio at each step of the training.
        """
        self.env.reset()
    
        ratios = []
        rewards, costs = deque(maxlen=self.width), deque(maxlen=self.width)
    
        for step in range(self.n_steps):
            # First, process the agent and the environment
            action = self.agent.sample_action(self.env.state)
            next_state, (reward, cost), _, _ = self.env.step(action)
            self.agent.update((reward, cost), next_state)

            # Next, process the rewards and costs signals
            rewards.append(reward)
            costs.append(cost)
            ratios.append(np.mean(rewards) / np.mean(costs))
    
            if step % self.print_interval == 0:  # I/O callbacks
                if self.stdouting:
                    self.stdout_callback(step=step, ratio=ratios[-1])
    
                if self.logging:
                    self.logger_callback(ratios=ratios)
    
        
        if self.logging:  # Final I/O callback
            self.logger_callback(ratios=ratios)

        if self.plotting:  # Plotting callback
            self.plot_callback(ratios=ratios)

        return ratios


# Skeletons for eventual ExperimentRunner class

class ConfigGenerator:
    """
    Generates configuration dictionaries that define the experiment
    ExperimentRunner is to carry out.

    Stores an experiment specification, which will likely be either a function
    or file. Generates or returns a list of tuples
        (env_config, agent_config, iomanager_config)
    containing all information needed to create a corresponding TrialRunner.
    """
    def __init__(self, experiment_spec):
        raise NotImplementedError
    
    def generate_configs(self):
        """
        Return experiment_configs list or generator.
        """
        raise NotImplementedError

class ExperimentRunner:
    """
    Overall coordinator of the experiment specified by ConfigGenerator.

    The only Ray objects that will be used are remote actor versions
    of TrialRunners.
    """
    def __init__(self):
        self.trial_constructor = TrialConstructor()
        self.trial_coordinator = TrialCoordinator()
        self.ray_controller = RayController()
        self.experiment_configs = None
        self.resource_configs = None
        raise NotImplementedError

    def register_configs(self, experiment_configs):
        """
        Parse experiment_configs for use by TrialConstructor.
        """
        raise NotImplementedError

    def register_resources(self, resource_configs):
        """
        Parse resource_configs for use by RayController.
        """
        raise NotImplementedError

    def provision_resources(self):
        """
        Make sure we have enough resources to run all trials in parallel with
        the desired number of cpus and gpus per trial.

        If not, either raise an error or enforce alternative provisioning.

        This must be called before TrialConstructor constructs TrialRunners!
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Use RayController, TrialConstructor, and TrialCoordinator to set up
        and run the experiment.
        """
        raise NotImplementedError


class RayController:
    """
    Keeps track of available compute resources. Starts and stops Ray.
    """
    def __init__(self):
        self.num_cpus = None
        self.num_gpus = None
        self.cpus_per_task = None
        self.gpus_per_task = None
        raise NotImplementedError

    def register_resources(self, resources):
        raise NotImplementedError

    def start_ray(self):
        raise NotImplementedError

    def stop_ray(self):
        raise NotImplementedError


class TrialConstructor:
    """
    Using the experiment_configs parsed by ExperimentRunner, creates
    corresponding TrialRunners to be handed off to TrialCoordinator.
    """
    def __init__(self):
        self.env_constructor = EnvConstructor()
        self.agent_constructor = AgentConstructor()
        self.iomanager_constructor = IOManagerConstructor()
        self.experiment_configs = None
        raise NotImplementedError

    def register_configs(self, experiment_configs):
        """
        Register experiment_configs for use in creating TrialRunners.
        """
        raise NotImplementedError

    def create_trials(self):
        """
        Create TrialRunners and return a list of their 
        """
        raise NotImplementedError


class EnvConstructor:
    """
    Constructs environments.
    """
    def __init__(self):
        raise NotImplementedError

    def create_env(self, env_config):
        raise NotImplementedError


class AgentConstructor:
    """
    Constructs agents.
    """
    def __init__(self):
        raise NotImplementedError

    def create_agent(self, agent_config):
        raise NotImplementedError


class IOManagerConstructor:
    """
    Constructs IOManagers.
    """
    def __init__(self):
        raise NotImplementedError

    def create_iomanager(self, iomanager_config):
        raise NotImplementedError


class TrialCoordinator:
    """
    Coordinates parallelization and execution of trials.
    """
    def __init__(self):
        self.trials = None
        self.resources = None
        raise NotImplementedError

    def gather_trials(self, trials):
        """
        Store list of TrialRunners to be run.
        """
        raise NotImplementedError

    def check_io_conflicts(self):
        """
        Check for conflicting IOManager output directories. We don't want
        multiple TrialRunners trying to write to the same directory.
        """
        raise NotImplementedError

    def register_resources(self, resource_configs):
        """
        Take note of the resources Ray is to be initialized with.
        """
        raise NotImplementedError

    def launch_trials(self):
        """
        Launch the experiment.
        """
        raise NotImplementedError