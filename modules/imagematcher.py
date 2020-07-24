import logging
from enum import Enum

import cv2
import numpy as np

_L = logging.getLogger('ImageMatcher')


class Column(Enum):
    CNN_DESC = 'cnn_descs'
    KEYLINE = 'keylines'


class ImageMatcher:
    def run(self, max_distance: float = -1.0, count_gt: bool = False) -> None:
        # compute best CNN matches from left to right
        cnn_matches = self.__compute_cnn_matches(count_gt)

        # debug information
        _L.debug('maximum distance used for visualization: {}'.format(max_distance))

        # visualize computed CNN matches
        self.__visualize_matches(cnn_matches, max_distance, 'CNN')

    def __init__(self, image_left: np.ndarray = None, npz_left: np.lib.npyio.NpzFile = None,
                 image_right: np.ndarray = None, npz_right: np.lib.npyio.NpzFile = None) -> None:
        # check if all parameters are set
        if any(par is None for par in [image_left, npz_left, image_right, npz_right]):
            _L.critical('At least one of the given parameters is None!')
            exit(1)

        # check if all variables are available
        for npz in [npz_left, npz_right]:
            if any(key not in npz for key in
                   [Column.CNN_DESC.value, Column.KEYLINE.value]):
                _L.critical('Could not find all required variables in \'{}\'!'.format(npz.fid.name))
                exit(1)

        # set instance variables
        self.__image_left = image_left
        self.__image_right = image_right
        self.__npz_left = npz_left
        self.__npz_right = npz_right

        # debug information
        _L.debug('shape of image_left: {}'.format(self.__image_left.shape))
        _L.debug('npz_left contains {} CNN descriptors'.format(len(self.__get_cnn_descriptors(npz_left))))
        _L.debug('shape of image_right: {}'.format(self.__image_right.shape))
        _L.debug('npz_right contains {} CNN descriptors'.format(len(self.__get_cnn_descriptors(npz_right))))

    @staticmethod
    def __get_cnn_descriptors(npz: np.lib.npyio.NpzFile) -> list:
        return npz[Column.CNN_DESC.value]

    @staticmethod
    def __get_lines(npz: np.lib.npyio.NpzFile) -> list:
        return npz[Column.KEYLINE.value]

    @staticmethod
    def __pairwise_cnn_distance(descs: np.ndarray) -> np.ndarray:
        # from pairwise_distance in metric_loss_ops.py in tensorflow
        # math background: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow

        sq = np.square(descs)
        reduced_sum_1 = np.add.reduce(np.transpose(sq))[:, None]
        reduced_sum_2 = np.transpose(np.add.reduce(np.transpose(sq))[:, None])
        reduced_added_sums = reduced_sum_1 + reduced_sum_2

        multiplied_featureMat = np.matmul(descs, np.transpose(descs))

        pairwise_distances_squared = reduced_added_sums - 2.0 * multiplied_featureMat
        pairwise_distances_squared = pairwise_distances_squared.clip(min=0)

        error_mask = np.less_equal(pairwise_distances_squared, 0.0)

        # This is probably not necessary, as we use squared distances here
        pairwise_distances_squared = np.multiply(pairwise_distances_squared,
                                                 np.logical_not(error_mask).astype(np.float32))

        num_data = pairwise_distances_squared.shape[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = np.ones_like(pairwise_distances_squared) - np.diag(np.ones([num_data]))

        pairwise_distances_squared = np.multiply(pairwise_distances_squared, mask_offdiagonals)

        return pairwise_distances_squared

    def __compute_cnn_matches(self, count_gt: bool = False) -> list:
        descriptors_left = np.asarray(ImageMatcher.__get_cnn_descriptors(self.__npz_left))
        descriptors_right = np.asarray(ImageMatcher.__get_cnn_descriptors(self.__npz_right))
        return ImageMatcher.__extract_matches(descriptors_left, descriptors_right, ImageMatcher.__pairwise_cnn_distance,
                                              count_gt)

    @staticmethod
    def __extract_matches(descriptors_left: np.ndarray, descriptors_right: np.ndarray, pairwise_distance: any,
                          count_gt: bool = False) -> list:
        # concatenate arrays
        descriptors = np.concatenate((descriptors_left, descriptors_right), axis=0)

        # compute pairwise distance matrix of descriptors
        distance = pairwise_distance(descriptors)

        # extract ROI from distance matrix
        indices_left = len(descriptors_left)
        indices_right = len(descriptors_right)
        region_of_interest = distance[0:indices_left, indices_left:indices_left + indices_right]
        best_matches = region_of_interest.argmin(axis=1)

        # extract matches as tuples (<index_left>, <index_right>, <distance>)
        correct_matches = 0
        false_matches = 0
        matches = []
        for index_left in range(len(best_matches)):
            index_right = best_matches[index_left]
            distance = region_of_interest[index_left, index_right]
            matches.append((index_left, index_right, float(distance)))
            if index_right == index_left:
                correct_matches += 1
            else:
                false_matches += 1

        if count_gt :
            print("Correct Matches: {} false Matches: {}".format(correct_matches, false_matches))

        # Visualize GT to check for correctness
        #        for index_left in range(len(best_matches)):
        #            index_right = index_left #best_matches[index_left]
        #            distance = region_of_interest[index_left, index_right]
        #            matches.append((index_left, index_right, float(distance)))

        return matches

    def __visualize_matches(self, matches: list, max_distance: float, title: str) -> None:
        lines_left = ImageMatcher.__get_lines(self.__npz_left)
        lines_right = ImageMatcher.__get_lines(self.__npz_right)

        # copy images from instance
        image_left = self.__image_left.copy()
        image_right = self.__image_right.copy()

        # if the images have a different size adjust scale of right image
        scale = 1.0
        if image_left.shape != image_right.shape:
            scale = image_left.shape[0] / image_right.shape[0]
            shape = (int(np.floor(image_right.shape[1] * scale)), image_left.shape[0])
            image_right = cv2.resize(image_right, shape, interpolation=cv2.INTER_LINEAR)

        # horizontally concatenate supplied images
        image = np.concatenate((image_left, image_right), axis=1)

        # compute aspect ratio of image
        aspect_ratio = image.shape[1] / image.shape[0]

        # get horizontal offset caused by concatenation
        horizontal_offset = image_left.shape[1]

        # count of skipped matches
        skipped = 0

        # iterate through all computed matches and visualize matching lines
        # tuples_left = []
        # tuples_right = []
        tuples_center = []
        for i in range(len(matches)):
            # matches are tuples (<index_left>, <index_right>, <distance>)
            index_left = matches[i][0]
            index_right = matches[i][1]
            distance = matches[i][2]

            # skip matches where the distance is larger than given threshold
            if 0 < max_distance < distance:
                skipped += 1
                continue

            # get copies of lines (instead of references)
            line_left = lines_left[index_left].copy()
            line_right = lines_right[index_right].copy() * scale  # don't forget to scale lines as well

            # add horizontal offset to x values of right line
            line_right[0][0] += horizontal_offset
            line_right[1][0] += horizontal_offset

            # compute center points of lines
            center_left = (line_left[0] + line_left[1]) * .5
            center_right = (line_right[0] + line_right[1]) * .5

            # save tuples of matched lines
            # tuples_left.append([tuple(line_left[0]), tuple(line_left[1])])
            # tuples_right.append([tuple(line_right[0]), tuple(line_right[1])])
            tuples_center.append([tuple(center_left), tuple(center_right)])

        # visualize left lines
        for i in range(len(lines_left)):
            start_point = lines_left[i][0].copy()
            end_point = lines_left[i][1].copy()
            image = cv2.line(image, tuple(start_point), tuple(end_point), (255, 0, 0), 8, cv2.LINE_AA)

        # visualize right lines
        for i in range(len(lines_right)):
            start_point = lines_right[i][0].copy()
            end_point = lines_right[i][1].copy()
            start_point[0] += horizontal_offset
            end_point[0] += horizontal_offset
            image = cv2.line(image, tuple(start_point), tuple(end_point), (255, 0, 0), 8, cv2.LINE_AA)

        # visualize matches
        for i in range(len(tuples_center)):
            tuple_center = tuples_center[i]
            image = cv2.line(image, tuple_center[0], tuple_center[1], (0, 255, 0), 2, cv2.LINE_AA)

        # debug information
        _L.debug('right image was scaled by factor {}'.format(scale))
        _L.debug('visualized {} matches ({} were skipped)'.format(len(matches) - skipped, skipped))

        # compute window size based on supplied images
        window_width = 1894
        window_height = int(np.floor(window_width * (1 / aspect_ratio)))

        # set window parameters
        window_name = '{} - ImageMatcher'.format(title)
        window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED

        # show final image
        cv2.namedWindow(window_name, window_flags)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
