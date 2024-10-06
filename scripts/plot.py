import argparse

import cv2
import numpy as np
from tqdm import tqdm
from pandas import read_csv
from feat import Detector
from feat.plotting import plot_face
from feat.plotting import plot_face
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(description="Plot face detection results from a csv file.")
    parser.add_argument("input_path", type=str, help="Path to input csv file")
    return parser.parse_args()


def main():
    args = parse_args()
    csv = read_csv(args.input_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    img_size = (640, 480)
    out = cv2.VideoWriter("output.mp4", fourcc, 30.0, img_size)

    aus = csv[csv.columns[1:21]]
    # for i, frame in enumerate(aus.values):
    for i, frame in tqdm(enumerate(aus.values), total=aus.shape[0]):
        # Create the face plot
        plot_face(au=frame)
        name = f"temp_{i:04d}.png"
        plt.draw()
        plt.savefig(name)
        plt.close()

        img = cv2.imread(name)
        img = cv2.resize(img, img_size)
        out.write(img)

    out.release()

if __name__ == "__main__":
    main()
