import argparse
import importlib

import costaware.utils.experiment as experiment
from costaware.core.envs import RandomMDPEnv
from costaware.core.agents import TabularQAgent


parser = argparse.ArgumentParser(
    'python experiment_runner.py'
)

parser.add_argument('--num_trials', type=int, default=2,
                    help='Number of trials to run')
parser.add_argument('--cpus_per_trial', type=int, default=1,
                    help='Number of CPUs to allocate to each trial')
parser.add_argument('--output_dir', type=str,
                    default='experiment_runner_data',
                    help='Directory to store trial data in')

args = parser.parse_args()

if __name__ == '__main__':
    output_dirs = [f'{args.output_dir}_{i}' for i in range(args.num_trials)]

    env_config = {'class': RandomMDPEnv,
                  'args': [5, 5, 'r1', 'c1'],
                  'kwargs': {'transition_seed': 1066}}
    env = experiment.EnvConstructor().create(env_config)

    agent_config = {'class': TabularQAgent,
                    'args': [env.states, env.actions, 0.01, 0.001],
                    'kwargs': dict()}

    agent_name = agent_config["class"].__name__

    iomanager_configs = [{'class': experiment.IOManager,
                          'args': [output_dir],
                          'kwargs': {
                              'print_interval': 1000,
                              'log_interval': 1000,
                              'agent_name': agent_name,
                              'filename': 'ratios',
                          }} for output_dir in output_dirs]

    trial_config = {'width': 100, 'n_steps': 10_000, 'n_episodes': 10,
                    'log': True, 'plot': False, 'print': True}

    # TODO Pass the actual all_configs_dict instead of an empty dict
    config_tuples = [experiment.ConfigTuple(
            env_config, agent_config, iomanager_config, trial_config, {}
        ) for iomanager_config in iomanager_configs]

    ray_configs = {'num_cpus': args.num_trials*args.cpus_per_trial,
                   'num_gpus': 0,
                   'cpus_per_trial': args.cpus_per_trial,
                   'gpus_per_trial': 0}

    expr = experiment.ExperimentRunner()
    expr.register_experiment_configs(config_tuples)
    expr.register_ray_configs(ray_configs)
    expr.verify_configs()
    expr.initialize_ray()
    expr.run_experiment()
    expr.shutdown_ray()
