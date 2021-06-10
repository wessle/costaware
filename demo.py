import argparse
import os

from examples.deep_Q_runner import run_deep_Q
from examples.synthetic_runner import run_synthetic
from examples.plotter import plot


parser = argparse.ArgumentParser(
    'python demo.py'
)
parser.add_argument('demo_type', type=str,
                    default=None, choices=['synthetic', 'deep_Q'],
                    help="Choose type of demo to run: 'synthetic' or 'deep_Q'")

parser.add_argument('--title', type=str,
                    default='title', 
                    help="Title of the plot")

parser.add_argument('--xlabel', type=str,
                    default='steps',
                    help="Label for the plot's x-axis")

parser.add_argument('--ylabel', type=str,
                    default='ratios',
                    help="Label for the plot's y-axis")

parser.add_argument('--confidence', type=int,
                    default=95,
                    help="Confidence level (1-99) for the confidence intervals calculated on the plot")

parser.add_argument('--filename', type=str,
                    default='comparison',
                    help='Name (excluding extension) of the generated plot and csv data file.')

parser.add_argument('--ext', type=str,
                    default='png',
                    help='Extension (excluding leading \'.\') of the plot file. Note: only matplotlib-supported extensions are valid.')



if __name__ == '__main__':
    args = parser.parse_args()
    
    output_dir = os.path.join('data', args.demo_type)
    config = os.path.join('examples', args.demo_type + '_config.yml')

    runner = run_synthetic if args.demo_type == 'synthetic' else run_deep_Q
    runner(output_dir, config)

    plot(
        output_dir,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        filename=args.filename,
        ext=args.ext,
        confidence=args.confidence
    )
