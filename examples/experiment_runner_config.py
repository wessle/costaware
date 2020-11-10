import argparse
import yaml
from importlib import import_module


import main.utils.experiment as experiment


parser = argparse.ArgumentParser(
    'python experiment_runner_config_example.py'
)

parser.add_argument('--num_trials', type=int, default=2,
                    help='Number of trials to run')
parser.add_argument('--cpus_per_trial', type=int, default=1,
                    help='Number of CPUs to allocate to each trial')
parser.add_argument('--output_dir', type=str,
                    default='experiment_runner_example',
                    help='Directory to store trial data in')
parser.add_argument('--config', type=str, default='config.yml',
                    help='Filename of YAML file containing trial configs')

args = parser.parse_args()


if __name__ == '__main__':

    # read in YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # check if there's an output_dir specified in the config file
    output_dir = config['output_dir'] if config['output_dir'] is not None \
            else args.output_dir
    
    # make list of output_dirs, one for each trial replication
    output_dirs = [f'{output_dir}_{i}' \
                   for i in range(args.num_trials)]


    # load the module where the envs are stored
    envs_module = import_module(config['envs_module_name'])

    # overwrite env class name with actual env class
    env_config = config['env_config']
    env_config['class'] = getattr(envs_module, env_config['class'])

    # create env
    env = experiment.EnvConstructor().create(env_config)


    # load the module where the agents are stored
    agents_module = import_module(config['agents_module_name'])

    # overwrite agent class name with actual agent class
    agent_config = config['agent_config']
    agent_config['class'] = getattr(agents_module, agent_config['class'])

    # insert environment-specific info into agent args
    agent_config['args'] = [env.states, env.actions] + agent_config['args']

    
    # make IOManagers for each env
    iomanager_configs = [
        {'class': experiment.IOManager,
         'args': [output_dir],
         'kwargs': dict()} \
        for output_dir in output_dirs]


    # copy trial_config
    trial_config = config['trial_config']

    # create a list of ConfigTuples, one for each trial replication
    config_tuples = [experiment.ConfigTuple(
        env_config, agent_config, iomanager_config, trial_config) \
        for iomanager_config in iomanager_configs]

    # create the ray configs
    ray_configs = {'num_cpus': args.num_trials*args.cpus_per_trial, 'num_gpus': 0,
                   'cpus_per_trial': args.cpus_per_trial, 'gpus_per_trial': 0}


    # run the experiment
    expr = experiment.ExperimentRunner()
    expr.register_experiment_configs(config_tuples)
    expr.register_ray_configs(ray_configs)
    expr.verify_configs()
    expr.initialize_ray()
    expr.run_experiment()
    expr.shutdown_ray()
