import os 
import yaml

import numpy as np
import ray
import matplotlib.pyplot as plt
from collections import deque
from itertools import product


import main.core.envs as envs
import main.utils.defaults as defaults


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
           Whether the experi)ent will save ratios to a file.
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
        self.experiment_spec = experiment_spec
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
        self.experiment_configs = None
        self.ray_configs = None
        self.ray_controller = RayController(self)

    def register_experiment_configs(self, experiment_configs):
        """
        Parse and store experiment_configs.

        For now experiment_configs is just a list of tuples of the form
            (env_config, agent_config, iomanager_config),
        where each entry is a dictionary, and each tuple completely specifies
        a trial to be run.
        """
        self.experiment_configs = experiment_configs

    def register_ray_configs(self, ray_configs):
        """
        Parse and store ray_configs.

        For now ray_configs is simply a dictionary of the form
            {'num_cpus': int,
             'num_gpus': int,
             'cpus_per_trial': int,
             'gpus_per_trial': int}
        num_cpus and num_gpus are needed when starting Ray in RayController,
        while cpus_per_trial and gpus_per_trial are needed when defining
        RayTrialRunner inside TrialConstructor. All four values are needed
        when checking whether num_cpus and num_gpus are sufficient for
        cpus_per_trial and gpus_per_trial, given the number of trials
        specified in experiment_configs.
        """
        self.ray_configs = ray_configs
        self.__dict__.update(self.ray_configs)

    def verify_configs(self):
        """
        Make sure we have enough resources to run all trials in parallel with
        the desired number of cpus and gpus per trial. If not, raise an error.

        Check inside experiment_configs to ensure that no two TrialRunners
        will attempt to write to the same directory. If a conflict is found,
        raise an error.

        This must be called before initialize_ray() and run_experiment()!
        """
        sufficient_cpus = len(self.experiment_configs) * self.cpus_per_trial \
                <= self.num_cpus
        sufficient_gpus = len(self.experiment_configs) * self.cgus_per_trial \
                <= self.num_cgus

        assert sufficient_cpus and sufficient_gpus, 'Not enough resources.'
        
        output_dirs = [trial_tuple[2]['output_dir'] for trial_tuple in \
                      self.experiment_configs]

        assert len(set(output_dirs)) == len(self.experiment_configs), \
                'Each trial must write to a distinct output directory.'


    def initialize_ray(self):
        """
        Initialize Ray with the specified ray_configs. If Ray is already running,
        first shut it down, then initialize.

        This must be called after verify_configs() and before run_experiment().
        """
        self.ray_controller.start_ray()

    def shutdown_ray(self):
        """
        Shut Ray down.
        """
        self.ray_controller.stop_ray()

    def run_experiment(self):
        """
        Run the experiment specified by the current experiment_configs. Ray must
        already be initialized with the desired ray_configs and verify_configs()
        should already have been called.

        First, a TrialConstructor and a TrialCoordinator are created. Next,
        TrialConstructor defines the RayTrialRunner object according to the
        ray_configs. It's necessary to define a new RayTrialRunner every time
        the experiment is run in order to allocate it the user-specified resources
        (e.g. num_cpus, num_gpus) contained in ray_configs. Third,
        TrialConstructor constructs a RayTrialRunner remote actor for each of the
        trials defined by experiment_configs. The Ray object IDs for the
        RayTrialRunners are then handed off to the TrialCoordinator, which
        stores them and launches them as Ray remote tasks. Once all tasks
        have been completed (which means all trials have been successfully
        run and logging data saved to disk), the function returns and it is
        safe to call shutdown_ray().

        Note that run_experiment() can be called multiple times over the life of
        ExperimentRunner, potentially on different experiment_configs and
        ray_configs. The only requirements are that Ray must be running and
        verify_configs should be called on the current experiment_configs and
        ray_configs to ensure they are compatible.
        """
        trial_constructor = TrialConstructor(self)
        trial_coordinator = TrialCoordinator(self)

        trial_constructor.define_ray_trial_runner()
        trials = trial_constructor.create_trials()

        trial_coordinator.gather_trials(trials)
        trial_coordinator.launch_trials()


class RayController:
    """
    Starts and stops Ray. Uses the ray_configs stored inside ExperimentRunner
    to decide how to initialize Ray.
    """
    def __init__(self, experiment_runner):
        self.experiment_runner = experiment_runner
        self.ray_running = False

    def _get_ray_init_configs(self):
        """
        Retrieve key-value pairs from ray_configs in EnvironmentRunner
        that must be passed to ray.init() in start_ray().
        """
        ray_configs = self.experiment_runner.ray_configs
        return {key: ray_configs[key] for key in ['num_cpus', 'num_gpus']}

    def start_ray(self):
        """
        Initialize Ray with the desired ray_configs in EnvironmentRunner.
        """
        ray.init(**self._get_ray_init_configs())
        self.ray_running = ray.is_initialized()
        assert self.ray_running, "Ray was not initialized for some reason!"

    def stop_ray(self):
        """
        Shut down Ray.
        """
        assert self.ray_running, "Ray must be running in order to be shut down"
        ray.shutdown()


class TrialConstructor:
    """
    Using the experiment_configs and ray_configs stored inside
    ExperimentRunner, creates corresponding RayTrialRunners to be handed off
    to TrialCoordinator.
    """
    def __init__(self, experiment_runner):
        self.experiment_runner = experiment_runner
        self.RayTrialRunner = None
        self.env_constructor = EnvConstructor()
        self.agent_constructor = AgentConstructor()
        self.iomanager_constructor = IOManagerConstructor()

    def _get_ray_actor_configs(self):
        """
        Retrieve key-value pairs from ray_configs in ExperimentRunner that
        need to be passed to @ray.remote in define_ray_trial_runner().
        """
        ray_configs = self.experiment_runner.ray_configs
        keys = ['num_cpus', 'num_gpus']
        vals = [ray_configs[key] for key in ['cpus_per_trial', 'gpus_per_trial']]
        return dict(zip(keys, vals))

    def define_ray_trial_runner(self):
        """
        Define the RayTrialRunner Ray Actor with the configuration
        (e.g. num_cpus, num_gpus) specified in ExperimentRunner's ray_configs.

        Must be called before create_trials().
        """
        @ray.remote(**self._get_ray_actor_configs())
        class RayTrialRunner(TrialRunner):
            def __init__(self, env, agent, io_manager, **kwargs):
                super().__init__(env, agent, io_manager, **kwargs)

        self.RayTrialRunner = RayTrialRunner

    def create_trials(self):
        """
        Create RayTrialRunners and return a list of their Ray object ids.

        Must be called after define_ray_trial_runner().
        """
        experiment_configs = self.experiment_runner.experiment_configs
        trials = []
        for trial_tuple in experiment_configs:
            env_config, agent_config, iomanager_config = trial_tuple
            env = self.env_constructor.create_env(env_config)
            agent = self.agent_constructor.create_agent(agent_config)
            iomanager = self.iomanager_constructor.create_iomanager(
                iomanager_config)
            trials.append(self.RayTrialRunner.remote(env, agent, iomanager))

        return trials


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
    Coordinates execution of trials.
    """
    def __init__(self, experiment_runner):
        self.experiment_runner = experiment_runner
        self.trials = None

    def gather_trials(self, trials):
        """
        Store list of RayTrialRunners to be run.
        """
        self.trials = trials

    def launch_trials(self):
        """
        Launch the experiment. Catch the return values and return them.

        This call is where most of the time will be spent during a call
        to ExperimentRunner.run_experiment(). ray.get() is blocking and
        will not return until all RayTrialRunners have finished executing.
        """
        return_vals = ray.get(self.trials)

        return return_vals
