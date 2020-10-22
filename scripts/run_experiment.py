import main.core.agents as agents

from main.utils.experiment import IOManager, ExperimentManager, generate_mdp_env

parser = argparse.ArgumentParser(
    description='Insert script description here'
)

parser.add_argument(
    'agent', type='str', nargs=1, choices=['ac', 'q'],
    help='Agent architecture to run'
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

AgentClass = {
    'ac': agents.LinearAC,
    'q':  agents.TabularQ
}

if __name__ = '__main__':

    agent = AgentClass[args.agent](**c)
    env   = generate_mdp_env()
    io    = IOManager(output_dir)

    manager = ExperimentManager(env, agent, io)
    manager.train()
