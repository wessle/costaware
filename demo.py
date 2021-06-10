import argparse
import os

from examples.deep_Q_runner import run_deep_Q
from examples.synthetic_runner import run_synthetic


parser = argparse.ArgumentParser(
    'python demo.py'
)
parser.add_argument('demo_type', type=str,
                    default=None,
                    help="Choose type of demo to run: 'synthetic' or 'deep_Q'")
args = parser.parse_args()


if __name__ == '__main__':
    
    assert args.demo_type in {'synthetic', 'deep_Q'}, "Specify a demo type!"

    output_dir = os.path.join('data', args.demo_type)
    config = os.path.join('examples', args.demo_type + '_config.yml')

    if args.demo_type == 'synthetic':
        run_synthetic(output_dir, config)
    else:
        run_deep_Q(output_dir, config)
