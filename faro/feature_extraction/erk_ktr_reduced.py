import numpy as np

from faro.feature_extraction.erk_ktr import FE_ErkKtr


class FE_ErkKtrReduced(FE_ErkKtr):
    """
    Feature extractor for ERK-KTR biosensor (reduced output).

    Same as FE_ErkKtr but drops individual intensity columns,
    keeping only the computed ratios (cnr, cnr_median) and area.
    """

    _DROP_COLUMNS = [
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
    ]

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table, masks = super().extract_features(labels, image, df_tracked, metadata)
        table = table.drop(columns=self._DROP_COLUMNS, errors="ignore")
        return table, masks
