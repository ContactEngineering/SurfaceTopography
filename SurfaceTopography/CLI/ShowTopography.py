import argparse

from matplotlib import pyplot as plt

from SurfaceTopography.IO import read_topography


def main():
    parser = argparse.ArgumentParser(
        prog="show-topography", description="Visualize a surface topography"
    )
    parser.add_argument('filename')
    args = parser.parse_args()

    t = read_topography(args.filename)
    t.plot()
    plt.show()
