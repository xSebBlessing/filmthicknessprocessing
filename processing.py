# path handling and script timing
import os
import json
import time

# matrix operations
import math

# trigonometry
import numpy as np

# visualisation
from matplotlib import pyplot as plt

# custom module for processing
from src import film_thickness_processing as proc


def px2mm(x):
    return x / 108 * 5


def mm2px(x):
     return x * 108 / 5


if __name__ == "__main__":
    # track script runtime
    start = time.time()

    # load all variable information
    config_path = ["cfg", "config.json"]
    with open(os.path.join(*config_path)) as file:
        cfg = json.load(file)

    # flag for plot output
    visualize = cfg["visualize"]

    # Vector of displacement direction for laser beam reflected at a plane parallel to the prism's top plane
    # enables correction of camera misalignment
    parallel_displacement_dir = np.array(cfg["parallel_displacement_direction"])

    # calculate angle of incidence within the water
    water_angle = proc.calculate_angle_of_incidence_film(cfg)

    # load images and names
    # first image in images is considered the reference image
    # bg is currently not used, could be useful for automatic calibration
    bg, images, im_names = proc.load_images(cfg)
    image_center = np.array((bg.shape[0] // 2, bg.shape[1] // 2))

    # find image pairs to compare (here only (0, 1))
    pairs = proc.get_pairs([i for i in range(len(images))])

    # calculate correlations of image pairs
    corr_maps = []
    for (i, j) in pairs:
        print(f"Corr {im_names[i]} and {im_names[j]}")
        corr_maps.append(proc.x_corr_ims(images[i], images[j]))

    # set up lists to store detected displacements
    parallel_displacements_mm = []
    print("Finding displacements")

    for corr_map, (i, j) in zip(corr_maps, pairs):
        print(f"Calculating displacement between {im_names[i]} and {im_names[j]}")
        displacement_start = time.time()

        # find maximum value in correlation map
        initial_peak_px = proc.find_correlation_peak(corr_map, visualize=visualize)

        # refine peak by fitting a paraboloid to it
        peak_px = proc.refine_peak_position(corr_map, initial_peak_px, visualize=visualize)

        # calculate displacement
        # correlation of the image with itself yields a peak in the image center
        # cross correlation peak minus image center yields displacement
        displacement_px = peak_px - image_center

        # convert to physical units with calibration
        displacement_mm = displacement_px / cfg["px_p_5mm"] * 5

        # find component in displacement direction
        parallel_displacement_mm = np.dot(parallel_displacement_dir, displacement_mm)

        # time and print results
        displacement_end = time.time()
        print(f"Finished in {displacement_end - displacement_start:.2f}s.")
        print(f"Laser beam was displaced by {parallel_displacement_mm:.2f}mm in y direction "
              f"between {im_names[i]} and {im_names[j]}.")

        # store results in lists for
        parallel_displacements_mm.append(parallel_displacement_mm)

    # calculate film thickness from beam displacement
    for displacement in [parallel_displacements_mm[-1]]:
        film_thickness = displacement / 2 / math.tan(water_angle.rad())
        print(f"Water film thickness is estimated to {film_thickness:.2f}mm.")

    # total runtime
    end = time.time()
    print(f"Total runtime: {end-start:.2f}s.")
