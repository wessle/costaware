import argparse
import yaml
from importlib import import_module
import os


import main.utils.experiment as experiment


parser = argparse.ArgumentParser(
    'python experiment_runner_experiment_config.py'
)
parser.add_argument('--output_dir', type=str,
                    default=None,
                    help='Directory to store trial data in')
parser.add_argument('--config', type=str,
                    default='experiment_config.yml',
                    help='Filename of YAML file containing trial configs')
args = parser.parse_args()


if __name__ == '__main__':

    with open(args.config, 'r') as f:
        experiment_config = yaml.safe_load(f)

    experiment_output_dir = args.output_dir if args.output_dir is not None \
            else experiment_config['output_dir']
    assert isinstance(experiment_output_dir, str), 'Specify a valid directory'

    ray_config = experiment_config['ray_config']

    trial_sets = experiment_config['trial_sets']


    # Create ConfigTuples for each trial specified in each trial set
    config_tuples = []
    for trial_set in trial_sets:

        config = trial_set['trial']

        # Create env
        envs_module = import_module(config['envs_module_name'])
        env_config = config['env_config']
        env_config['class'] = getattr(envs_module, env_config['class'])
        env = experiment.EnvConstructor().create(env_config)


        # Create agent
        agents_module = import_module(config['agents_module_name'])
        agent_config = config['agent_config']
        agent_config['class'] = getattr(agents_module, agent_config['class'])
        agent_config['args'] = [env.states, env.actions] + agent_config['args']

    
        # Set up all necessary directories for the IOManagers
        trial_set_dir = os.path.join(experiment_output_dir,
                                     trial_set['trial_set_name'])
        os.makedirs(trial_set_dir, exist_ok=True)
        fmt = os.path.join(
            trial_set_dir,
            config['iomanager_config']['args'][0]) + '_{}'
        output_dirs = [fmt.format(i) \
                       for i in range(trial_set['num_replications'])]

        # make IOManagers for each trial
        iomanager_config = config['iomanager_config']
        iomanager_config['class'] = experiment.IOManager
        iomanager_config['kwargs'].update({
            'agent_name': f'{agent_config["class"].__name__}'})
        iomanager_configs = []
        for output_dir in output_dirs:
            iomanager_configs.append(iomanager_config.copy())
            iomanager_configs[-1]['args'] = [output_dir]


        # Copy trial_config
        trial_config = config['trial_config']

        # Create a list of ConfigTuples, one for each trial replication
        config_tuples.extend(experiment.ConfigTuple(
            env_config, agent_config, iomanager_config, trial_config) \
            for iomanager_config in iomanager_configs)


    # run the experiment
    expr = experiment.ExperimentRunner()
    expr.register_experiment_configs(config_tuples)
    expr.register_ray_configs(ray_config)
    expr.verify_configs()
    expr.initialize_ray()
    expr.run_experiment()
    expr.shutdown_ray()
