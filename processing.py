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


if __name__ == "__main__":
    # track script runtime
    start = time.time()

    # load all variable information
    config_path = ["cfg", "config.json"]
    with open(os.path.join(*config_path)) as file:
        cfg = json.load(file)

    # flag for plot output
    visualise = cfg["visualise"]

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
    corrs = []
    for (i, j) in pairs:
        print(f"Corr {im_names[i]} and {im_names[j]}")
        corrs.append(proc.x_corr_ims(images[i], images[j]))

    # set up lists to store detected displacements
    displacements_mm = []
    print("Finding displacements")

    for corr, (i, j) in zip(corrs, pairs):
        print(f"Calculating displacement between {im_names[i]} and {im_names[j]}")
        displacement_start = time.time()

        # find maximum value in correlation map
        initial_peak_px = np.array(np.unravel_index(np.argmax(corr), corr.shape))

        # refine peak by fitting a paraboloid to it
        peak_px = proc.refine_peak_position(corr, initial_peak_px, vis=visualise)

        # calculate displacement
        # correlation of the image with itself yields a peak in the image center
        # cross correlation peak minus image center yields displacement
        displacement_px = peak_px - image_center

        # convert to physical units with calibration
        displacement_mm = displacement_px / cfg["px_p_5mm"] * 5

        # time and print results
        displacement_end = time.time()
        print(f"Finished in {displacement_end - displacement_start:.2f}s.")
        print(f"Laser beam was displaced by {displacement_px[0]:.2f}px or {displacement_mm[0]:.2f}mm in y direction "
              f"between {im_names[i]} and {im_names[j]}.")

        # store results in lists for
        displacements_mm.append(displacement_mm)

    # calculate film thickness from beam displacement
    for displacement in [displacements_mm[-1]]:
        film_thickness = np.linalg.norm(displacement) / 2 / math.tan(water_angle.rad())
        print(f"Water film thickness is estimated to {film_thickness:.2f}mm.")

    # total runtime
    end = time.time()
    print(f"Total runtime: {end-start:.2f}s.")

    # visualise results
    if visualise:
        # create image with all laser beam imprints
        cumulated = np.array(images[0])
        for image in images[1:]:
            cumulated += image

        # clip at 1 to maintain image format
        cumulated = np.clip(cumulated, 0, 1)

        # detect original object center

        # plot cumulated image with displacement vector
        fig, ax = plt.subplots()
        ax.imshow(cumulated, cmap="gray")
        plt.show()
