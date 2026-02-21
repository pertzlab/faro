import numpy as np
import pandas as pd

from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table

from rtm_pymmcore.feature_extraction.base import FeatureExtractor
from rtm_pymmcore.feature_extraction.utils import median_intensity


class FE_ErkKtrReduced(FeatureExtractor):
    """
    Feature extractor for ERK-KTR biosensor.
    This class implements a feature extractor that extracts features from the
    ERK-KTR biosensor images. It creates a cytosolic ring around the nucleus and
    extracts mean intensities from both the nucleus and the ring.
    """

    def __init__(self, used_mask, margin=2, distance=4):
        self.used_mask = used_mask
        self.name_extra_mask = "labels_ring"
        self.extra_folders = [self.name_extra_mask]
        self.margin = margin
        self.distance = distance
        super().__init__()

    def extract_ring(
        self,
        labels,
    ):  # distance = 10 for 40x; 4px for 20x
        """Create the cytosolic rings for biosensor dependant on nuclear/cytosolic fluorescence intensity.
        Args:
            margin: nb pixels between nucleus and ring
            distance: nb pixels ring width (margin is subtracted)
        """
        labels_expanded_margin = expand_labels(labels, distance=self.margin)
        labels_expanded_rings = expand_labels(labels, distance=self.distance)
        labels_expanded_rings[labels_expanded_margin != 0] = 0
        return labels_expanded_rings.astype(int)

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        """Create a table with features for every detected cell.
        Args:
            labels: frame with labeled nuclei
            image: raw frame with dimensions [C,X,Y]
            df_tracked: tracked dataframe (not used yet, available for future track-based features)
            metadata: frame metadata (not used yet)
        """
        image = np.moveaxis(image, 0, 2)  # CXY to XYC
        labels = labels[self.used_mask]

        labels_ring = self.extract_ring(labels)  # create cytosolic rings
        # EXTRACT FEATURES
        table_nuc = regionprops_table(
            labels,
            image,
            ["mean_intensity", "label", "area"],
            extra_properties=[median_intensity],
        )

        table_ring = regionprops_table(
            labels_ring,
            image,
            ["mean_intensity", "label"],
            extra_properties=[median_intensity],
        )

        # CREATE TABLES
        table_nuc = pd.DataFrame.from_dict(table_nuc)
        table_ring = pd.DataFrame.from_dict(table_ring)
        table_nuc = table_nuc.rename(
            {
                "mean_intensity-0": "mean_intensity_C0_nuc",
                "mean_intensity-1": "mean_intensity_C1_nuc",
                "mean_intensity-2": "mean_intensity_C2_nuc",
                "median_intensity-0": "median_intensity_C0_nuc",
                "median_intensity-1": "median_intensity_C1_nuc",
                "area": "area_nuc",
            },
            axis="columns",
        )
        table_ring = table_ring.rename(
            {
                "mean_intensity-0": "mean_intensity_C0_ring",
                "mean_intensity-1": "mean_intensity_C1_ring",
                "mean_intensity-2": "mean_intensity_C2_ring",
                "median_intensity-0": "median_intensity_C0_ring",
                "median_intensity-1": "median_intensity_C1_ring",
            },
            axis="columns",
        )

        # CONCAT TABLES
        table = table_nuc.merge(table_ring, on=["label"])

        # CALCULATE the ERK ratio
        table["cnr"] = table["mean_intensity_C1_ring"] / table["mean_intensity_C1_nuc"]
        table["cnr_median"] = (
            table["median_intensity_C1_ring"] / table["median_intensity_C1_nuc"]
        )

        table = table.drop(
            columns=[
                "mean_intensity_C0_nuc",
                "mean_intensity_C1_nuc",
                "mean_intensity_C2_nuc",
                "median_intensity_C0_nuc",
                "median_intensity_C1_nuc",
                "mean_intensity_C0_ring",
                "mean_intensity_C1_ring",
                "mean_intensity_C2_ring",
                "median_intensity_C0_ring",
                "median_intensity_C1_ring",
            ],
            errors="ignore",
        )

        table["cnr"] = table["cnr"].astype(np.float32)
        table["cnr_median"] = table["cnr_median"].astype(np.float32)
        table["area_nuc"] = table["area_nuc"].astype(np.float32)

        return table, [{self.name_extra_mask: labels_ring}]
