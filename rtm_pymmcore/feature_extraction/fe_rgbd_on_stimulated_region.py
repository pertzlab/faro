import numpy as np
import skimage
import pandas as pd
from .abstract_fe import FeatureExtractor


class FErGBD_stimulated_region(FeatureExtractor):
    def __init__(self, used_mask):
        self.used_mask = used_mask
        super().__init__()

    def extract_features(self, labels, image, stim_mask = None):
        raw_channel = image[0]
        seg = labels[self.used_mask]
        if stim_mask is None:
            stim_mask = np.zeros_like(seg, dtype=bool)
        props = skimage.measure.regionprops(seg, intensity_image=raw_channel)

        # Create a dataframe to store results
        results = []

        for prop in props:
            # Get the label ID
            label_id = prop.label
            
            # Create a mask for this specific cell
            cell_mask = seg == label_id
            
            # Compute stimulated area for this cell
            cell_stim_mask = np.logical_and(cell_mask, stim_mask)
            
            # Calculate intensity metrics
            if np.any(cell_stim_mask):
                # Intensity in stimulated area
                stim_intensities = raw_channel[cell_stim_mask]
                stim_mean_intensity = np.mean(stim_intensities)
                stim_total_intensity = np.sum(stim_intensities)
                stim_area = np.sum(cell_stim_mask)
                
                # Intensity in non-stimulated area of the cell
                non_stim_mask = np.logical_and(cell_mask, ~stim_mask)
                if np.any(non_stim_mask):
                    non_stim_intensities = raw_channel[non_stim_mask]
                    non_stim_mean_intensity = np.mean(non_stim_intensities)
                    non_stim_total_intensity = np.sum(non_stim_intensities)
                    non_stim_area = np.sum(non_stim_mask)
                else:
                    non_stim_mean_intensity = 0
                    non_stim_total_intensity = 0
                    non_stim_area = 0
                
                # Calculate ratio of stimulated to non-stimulated intensity
                intensity_ratio = stim_mean_intensity / non_stim_mean_intensity if non_stim_area > 0 else np.nan
                
                # Fraction of cell area that's stimulated
                stim_fraction = stim_area / (stim_area + non_stim_area)
            else:
                # Cell is not stimulated at all
                stim_mean_intensity = 0
                stim_total_intensity = 0
                stim_area = 0
                non_stim_mean_intensity = np.mean(raw_channel[cell_mask])
                non_stim_total_intensity = np.sum(raw_channel[cell_mask])
                non_stim_area = np.sum(cell_mask)
                intensity_ratio = 0
                stim_fraction = 0
            
            # Cell-wide metrics
            cell_centroid = prop.centroid
            cell_area = prop.area
            cell_mean_intensity = prop.mean_intensity
            
            # Store results
            results.append({
                'label': label_id,
                'x': cell_centroid[0],
                'y': cell_centroid[1],
                'cell_area': cell_area,
                'cell_mean_intensity': cell_mean_intensity,
                'stim_area': stim_area,
                'stim_mean_intensity': stim_mean_intensity,
                'stim_total_intensity': stim_total_intensity,
                'non_stim_area': non_stim_area,
                'non_stim_mean_intensity': non_stim_mean_intensity, 
                'non_stim_total_intensity': non_stim_total_intensity,
                'stim_fraction': stim_fraction,
                'intensity_ratio': intensity_ratio,
                'is_stimulated': bool(stim_area > 0)
            })
        return pd.DataFrame(results), None