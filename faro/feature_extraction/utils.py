import numpy as np


def median_intensity(region_mask, intensity_image):

    return np.median(intensity_image[region_mask], axis=0)
