import main.utils.experiment as experiment
from main.core.envs import RandomMDPEnv
from main.core.agents import TabularQAgent


# TODO: Get this working! Currently experiencing error regarding
# EnvConstructor inside TrialConstructor.

output_dir = 'experiment_runner_test_data'

env_config = {'class': RandomMDPEnv,
              'args': [5, 5, 'r1', 'c1'],
              'kwargs':dict()}
env = experiment.EnvConstructor.create(env_config)

agent_config = {'class': TabularQAgent,
                'args': [env.states, env.actions, 0.01, 0.001],
                'kwargs': dict()}

iomanager_config = {'class': experiment.IOManager,
                    'args': [output_dir],
                    'kwargs': dict()}

trial_config = {'width': 100, 'print_interval': 10000, 'n_steps': 500000,
                'logging': True, 'plotting': False, 'stdouting': True}

config_tuple = experiment.ConfigTuple(env_config, agent_config,
                                      iomanager_config, trial_config)

ray_configs = {'num_cpus': 2, 'num_gpus': 0,
               'cpus_per_trial': 2, 'gpus_per_trial': 0}

expr = experiment.ExperimentRunner()
expr.register_experiment_configs([config_tuple])
expr.register_ray_configs(ray_configs)
expr.verify_configs()
expr.initialize_ray()
expr.run_experiment()
expr.shutdown_ray()
