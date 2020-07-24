import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from modules.imagematcher import ImageMatcher

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
_L = logging.getLogger('ImageMatcher')


def init_parser():
    # instantiate ArgumentParser object
    parser = argparse.ArgumentParser()

    # add arguments for left and right images (filepath)
    parser.add_argument('-l', '--image_left', default=None, type=str, required=True, dest='image_left_filepath')
    parser.add_argument('-r', '--image_right', default=None, type=str, required=True, dest='image_right_filepath')

    # add arguments for left and right npz containers (filepath)
    parser.add_argument('--npz_left', default=None, type=str, required=False, dest='npz_left_filepath')
    parser.add_argument('--npz_right', default=None, type=str, required=False, dest='npz_right_filepath')

    # add argument for distance threshold
    parser.add_argument('--max_dist', type=float, default=-1.0, dest='max_distance')

    # add argument for GroundTruth counting, requires matching lines to have the same indices
    parser.add_argument("--count_gt", action="store_true",
                        help=("Count correct matches, requires GT information with matching lines having the same indices."))

    return parser


def run():
    # initialize parser and parse arguments
    parser = init_parser()
    args = parser.parse_args()

    # instantiate image paths
    image_left_path = Path(args.image_left_filepath)
    image_right_path = Path(args.image_right_filepath)

    # helper function to check if given path is a valid file
    def is_file(path):
        if not Path.is_file(path):
            _L.critical('Specified path \'{}\' is not a valid file!'.format(path))
            exit(1)

    # check if image file paths are valid
    is_file(image_left_path)
    is_file(image_right_path)

    # set extension for NPZ files
    extension = '.npz'

    # derive npz file paths from image file paths if not specified
    if args.npz_left_filepath is None:
        args.npz_left_filepath = str(image_left_path.with_suffix(extension))

    if args.npz_right_filepath is None:
        args.npz_right_filepath = str(image_right_path.with_suffix(extension))

    # instantiate npz paths
    npz_left_path = Path(args.npz_left_filepath)
    npz_right_path = Path(args.npz_right_filepath)

    # check if npz file paths are valid
    is_file(npz_left_path)
    is_file(npz_right_path)

    # load images
    image_left = cv2.imread(str(image_left_path), cv2.IMREAD_COLOR)
    image_right = cv2.imread(str(image_right_path), cv2.IMREAD_COLOR)

    # load npz containers
    try:
        npz_left = np.load(str(npz_left_path))
        npz_right = np.load(str(npz_right_path))
    except IOError:
        _L.critical('At least one of the \'.npz\' containers could not be loaded!')
        exit(1)

    # debug information
    _L.debug('path to image_left: \'{}\''.format(str(image_left_path)))
    _L.debug('path to npz_left: \'{}\''.format(str(npz_left_path)))
    _L.debug('path to image_right: \'{}\''.format(str(image_right_path)))
    _L.debug('path to npz_right: \'{}\''.format(str(npz_right_path)))

    # instantiate ImageMatcher object
    im = ImageMatcher(image_left=image_left, image_right=image_right, npz_left=npz_left, npz_right=npz_right)  # noqa

    # run ImageMatcher application
    im.run(max_distance=args.max_distance, count_gt=args.count_gt)


if __name__ == '__main__':
    run()
