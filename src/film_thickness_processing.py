# imports for path handling and script timing
import os
import time

# for type hinting
from typing import Union

# image reading
from PIL import Image

# matrix operations
import numpy as np

# trigonometry
import math

# convolution operation
from scipy import signal

# least squares fit for subpixel accuracy
from scipy import optimize as opt

# visualisation
from matplotlib import pyplot as plt
from matplotlib import patches

# custom angle class for rad / deg handling
from . import Angle


def find_image_names(cfg: dict) -> tuple[str, str, list[str]]:
    """
    reads the given data directory and identifies the images to evaluate

    :param cfg: global config dictionary
    :return: names of bg, ref and meas images
    """
    bg_name = ""
    ref_name = ""
    meas_names = []

    data_path = os.path.join(*cfg["data_directory"])

    meas_ids = []
    for idx in cfg["image_indices"]["meas"]:
        meas_ids.append(str(idx).zfill(2))

    bg_index = str(cfg["image_indices"]["bg"]).zfill(2)
    ref_index = str(cfg["image_indices"]["ref"]).zfill(2)

    for file in os.listdir(data_path):
        # noinspection PyTypeChecker
        if file.startswith(bg_index):
            bg_name = str(os.path.join(data_path, file))
        elif file.startswith(ref_index):
            ref_name = str(os.path.join(data_path, file))
        elif file[:2] in meas_ids:
            meas_names.append(str(os.path.join(data_path, file)))

    return bg_name, ref_name, meas_names


def load_images(cfg: dict) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """
    loads the images with names specified in global config

    :param cfg: global config dictionary
    :return: normalized image files
    """
    vmax = pow(2, int(cfg["cam_dynamic_range"])) - 1

    bg_path, ref_path, meas_paths = find_image_names(cfg)

    bg_image = np.array(Image.open(bg_path), dtype="float64") / vmax
    ref_image = np.array(Image.open(ref_path), dtype="float64") / vmax

    meas_images = [ref_image]
    for path in meas_paths:
        meas_images.append(np.array(Image.open(path), dtype="float64") / vmax)

    image_names = ["ref"]
    image_names.extend([f"im{i+1}" for i in range(len(meas_images))])

    return bg_image, meas_images, image_names


def x_corr_ims(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    uses scipy's fftconvolve function to calculate a fast correlation between im1 and im2
    convolution and correlation only differ in the sign of the weighting function

    :param im1: image 1 to correlate
    :param im2: image 2 to correlate
    :return: correlation map
    """
    start = time.time()
    result = signal.fftconvolve(im1, im2[::-1, ::-1], mode="same")
    end = time.time()
    print(f"Finished in {round(end - start, 2)}s.")

    return result


def find_correlation_peak(corr_map: np.ndarray, visualize: bool = False):
    peak = np.array(np.unravel_index(np.argmax(corr_map), corr_map.shape))

    if visualize:
        image_center = np.array(corr_map.shape) // 2
        extent = [-image_center[1], image_center[1], image_center[0], -image_center[0]]

        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(corr_map, extent=extent)
        ax.scatter(peak[1]-image_center[1]+0.5, peak[0]-image_center[0]+0.5, c="red", marker="x")

        ax.set_title("Correlation map with initial peak")
        ax.set_xlabel(r"$\Delta$x [px]")
        ax.set_ylabel(r"$\Delta$y [px]")

        cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Correlation level", rotation=90, labelpad=12)

        plt.show()

    return peak


def paraboloid(xy: list[Union[list[int], np.ndarray], Union[list[int], np.ndarray]],
               x0: int, y0: int, a: float, b: float, c: float) -> np.ndarray:
    """
    calculates a paraboloid function of a 2d mesh grid based on arbitrary coefficients

    :param xy: x y mesh
    :param x0: x offset
    :param y0: y offset
    :param a: z offset
    :param b: x scaling
    :param c: y scaling
    :return: 1d representation of paraboloid function values
    """
    x, y = np.meshgrid(*xy)

    f = a + np.square(x0 - x) / b + np.square(y0 - y) / c

    return f.ravel()


def residual(params: list, xy: list[Union[list[int], np.ndarray], Union[list[int], np.ndarray]],
             data: np.ndarray) -> np.ndarray:
    """
    Calculates difference between paraboloid and given 2d data set

    :param params: fitting coefficients (see paraboloid)
    :param xy: x y mesh
    :param data: data to fit to
    :return: 1d representation of difference between paraboloid and data to fit to
    """
    return paraboloid(xy, *params) - data.ravel()


def refine_peak_position(corr_map: np.ndarray, initial_peak: np.ndarray,
                         k_size: int = 11, visualize: bool = False) -> np.ndarray:
    """
    Refines the correlation peak position to sub pixel accuracy by fitting a paraboloid to the correlation peak and
    evaluating its maximum

    :param corr_map: correlation map of the image pair
    :param initial_peak: initial absolute correlation peak position from maximum function in px
    :param k_size: kernel size around initial peak to consider; must be uneven
    :param visualize: visualization flag
    :return: refined absolute peak position in px
    """

    # check for uneven kernel size
    if k_size % 2 == 0:
        print("Use only uneven kernel sizes!")
        exit(-1)

    # define x and y range of correlation map to consider
    x0 = initial_peak[1]
    y0 = initial_peak[0]

    k_size_half = k_size // 2

    x = [x0 - k_size_half + i for i in range(k_size)]
    y = [y0 - k_size_half + i for i in range(k_size)]

    # extract data of kernel size around the initial correlation peak
    data = corr_map[y0-k_size_half:y0+k_size_half+1,
                    x0-k_size_half:x0+k_size_half+1]

    # define initial guess for paraboloid fit using the initial peak position and the initial correlation peak value
    initial_guess = [x0, y0, np.max(data), -5, -5]

    # noinspection PyTypeChecker
    # optimize parameters for minimal residuals between correlation data and paraboloid
    popt = opt.leastsq(residual, initial_guess, args=([x, y], data))

    # extract refined x and y
    x_ref = popt[0][0]
    y_ref = popt[0][1]

    if visualize:
        image_center = np.array(corr_map.shape) // 2

        x0_vis = x0 - image_center[1]
        y0_vis = y0 - image_center[0]

        x_ref_vis = x_ref - image_center[1]
        y_ref_vis = y_ref - image_center[0]

        # create mesh for fitting
        x_lin = np.linspace(x0_vis - k_size_half - 0.5, x0_vis + k_size_half + 0.5, 1000)
        y_lin = np.linspace(y0_vis - k_size_half - 0.5, y0_vis + k_size_half + 0.5, 1000)

        # compute optimal paraboloid
        params = [x_ref_vis, y_ref_vis, *popt[0][2:]]
        opt_fun = paraboloid([x_lin, y_lin], *params).reshape(1000, 1000)

        # plot initial peak, refined peak and refinement function over correlation map
        extent_ax1 = [-image_center[1], image_center[1], image_center[0], -image_center[0]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        im = ax1.imshow(corr_map, extent=extent_ax1, vmin=np.min(data))

        # rectangle for kernel
        rect = patches.Rectangle((x[0]-image_center[1], y[0]-image_center[0]), k_size, k_size, linewidth=1,
                                 edgecolor="r", facecolor="none")
        ax1.add_patch(rect)

        ax1.set_title("Correlation map with kernel")
        ax1.set_xlabel(r"$\Delta$x [px]")
        ax1.set_ylabel(r"$\Delta$y [px]")

        extent_ax2 = [x0_vis - k_size_half - 0.5, x0_vis + k_size_half + 0.5,
                      y0_vis + k_size_half + 0.5, y0_vis - k_size_half - 0.5]

        ax2.imshow(data.reshape(k_size, k_size), vmin=np.min(data), extent=extent_ax2)
        ax2.scatter(x0_vis, y0_vis, marker="x", c="red")
        ax2.scatter(x_ref_vis, y_ref_vis, marker="x", c="black")
        ax2.contour(x_lin, y_lin, opt_fun, colors="black")

        ax2.set_title("initial and refined peak")
        ax2.set_xlabel(r"$\Delta$x [px]")
        ax2.set_ylabel(r"$\Delta$y [px]")

        cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 0.02, ax2.get_position().height])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Correlation level", rotation=90, labelpad=12)

        plt.show()

    return np.array([y_ref, x_ref])


def calculate_angle_of_incidence_film(cfg: dict) -> Angle.Angle:
    """
    calculates the angle of incidence inside the film for thickness calculations

    :param cfg: global config dictionary
    :return: Angle of incidence inside film caused by refraction
    """
    # read prism angle from dictionary
    prism_angle = Angle.Angle(cfg["prism_angle"])

    # read refractive indices from dictionary
    n_trans = cfg["refractive_indices"][cfg["materials"]["transparent"]]
    n_film = cfg["refractive_indices"][cfg["materials"]["film"]]

    ratio_n = n_trans / n_film

    # calculate refracted angle from Snell's law
    return Angle.Angle(math.asin(ratio_n * math.sin(prism_angle.rad())), unit="rad")


def get_pairs(number_list: list[int]) -> list[tuple[int, int]]:
    """
    finds all unique permutation of the indexes supplied in the input list
    adapted from:
    https://stackoverflow.com/questions/70413515/get-all-unique-pairs-in-a-list-including-duplicates-in-python

    :param number_list: list of image list indices to pair
    :return: list of pairs
    """
    out = []

    for i in number_list:
        for j in number_list:
            if i == j or (i, j) in out or (j, i) in out:
                continue
            else:
                if i < j:
                    out.append((i, j))
                else:
                    out.append((j, i))

    return out
