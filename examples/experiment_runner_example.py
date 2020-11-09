import main.utils.experiment as experiment
from main.core.envs import RandomMDPEnv
from main.core.agents import TabularQAgent


NUM_TRIALS = 2
CPUS_PER_TRIAL = 2

if __name__ == '__main__':
    output_dirs = [f'../data/experiment_runner_example_data_{i}' \
                   for i in range(NUM_TRIALS)]

    env_config = {'class': RandomMDPEnv,
                  'args': [5, 5, 'r1', 'c1'],
                  'kwargs': {'transition_seed': 1066,
                             'training_seed': 1789}}
    env = experiment.EnvConstructor().create(env_config)

    agent_config = {'class': TabularQAgent,
                    'args': [env.states, env.actions, 0.01, 0.001],
                    'kwargs': dict()}

    iomanager_configs = [
        {'class': experiment.IOManager,
         'args': [output_dir],
         'kwargs': dict()} \
        for output_dir in output_dirs]

    trial_config = {'width': 100, 'print_interval': 10000, 'n_steps': 100000,
                    'logging': True, 'plotting': False, 'stdouting': True}

    config_tuples = [experiment.ConfigTuple(
        env_config, agent_config, iomanager_config, trial_config) \
        for iomanager_config in iomanager_configs]

    ray_configs = {'num_cpus': NUM_TRIALS*CPUS_PER_TRIAL, 'num_gpus': 0,
                   'cpus_per_trial': CPUS_PER_TRIAL, 'gpus_per_trial': 0}

    expr = experiment.ExperimentRunner()
    expr.register_experiment_configs(config_tuples)
    expr.register_ray_configs(ray_configs)
    expr.verify_configs()
    expr.initialize_ray()
    expr.run_experiment()
    expr.shutdown_ray()
