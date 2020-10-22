import argparse
import os
import wesutils

import main.core.agents as agents

from datetime import datetime
from main.utils.experiment import IOManager, TrialRunner, generate_mdp_env

parser = argparse.ArgumentParser(
    description='Insert script description here'
)

parser.add_argument(
    'agent', type=str, choices=['ac', 'q'],
    help='Agent architecture to run'
)

parser.add_argument(
    '--agent_config', type=str, default="",
    help='YAML filename that provides the agent parameters for the experiment'
)

parser.add_argument(
    '--env_config', type=str, default="configs/default_env_config.yml",
    help='YAML filename that provides the environment parameters for the experiment'
)

parser.add_argument(
    '--output', type=str, default='data',
    help='top-level directory name that all experimental output will be saved'
)

args = parser.parse_args()

AgentClass = {
    'ac': agents.LinearACAgent,
    'q':  agents.TabularQAgent
}

if __name__ == '__main__':

    # Easiest thing to do right now is to separate the file for the environment
    # parameters from the file for agent parameters
    agent_config = wesutils.load_config(args.agent_config)
    env_config   = wesutils.load_config(args.env_config)

    states, actions, env = generate_mdp_env(**env_config)
    agent  = AgentClass[args.agent](states, actions, **agent_config)

    now = datetime.now()
    time_str = '_'.join([
        f'{now.year}{now.month}{now.day}',
        f'{now.hour:02d}{now.minute:02d}{now.second:02d}'
    ])

    output_dir = os.path.join(
        args.output,
        '_'.join([
            agent.title,
            f's{len(states)}a{len(actions)}',
            time_str
        ])
    )

    io = IOManager(output_dir)
    io.save_yml('agent_config.yml', agent_config)
    io.save_yml('env_config.yml', env_config)

    manager = TrialRunner(env, agent, io)
    manager.train()
