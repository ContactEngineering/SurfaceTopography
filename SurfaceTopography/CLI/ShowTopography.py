import argparse

from matplotlib import pyplot as plt

from SurfaceTopography.IO import read_topography


def main():
    parser = argparse.ArgumentParser(
        prog="show-topography", description="Visualize a surface topography"
    )
    parser.add_argument('filename')
    parser.add_argument(
        '--grid-points', '-g',
        action='store_true',
        help='Label axes by grid point indices instead of physical positions'
    )
    args = parser.parse_args()

    t = read_topography(args.filename)
    t.plot(axes_in_grid_points=args.grid_points)
    plt.show()
