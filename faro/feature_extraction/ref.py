import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import pandas as pd
from .base_ref import FeatureExtractorRef
from ..core.utils import labels_to_particles


class RefFE(FeatureExtractorRef):

    def extract_features(self, segmentation_results, image, df_tracked, metadata):
        tracked_label = labels_to_particles(
            segmentation_results[self.used_mask], df_tracked
        )
        tracked_label = np.expand_dims(tracked_label, 0)
        table = skimage.measure.regionprops_table(
            tracked_label, image, properties=["label", "mean_intensity"]
        )
        table = pd.DataFrame.from_dict(table)
        table = table.rename(
            {
                "mean_intensity": "ref_mean_intensity",
                "label": "particle",
            },
            axis=1,
        )
        table = pd.DataFrame.from_dict(table)

        if not self.multi_timepoint:
            df_tracked = df_tracked.merge(table, on=["particle"])
        elif self.multi_timepoint:
            if "ref_mean_intensity" not in df_tracked.columns:
                df_tracked["ref_mean_intensity"] = np.nan
            if "fov_timestep" in metadata:
                timestep_name_variable = "fov_timestep"
            else:
                timestep_name_variable = "timestep"

            table[timestep_name_variable] = metadata[timestep_name_variable]
            table["fov"] = metadata["fov"]
            # Update only matching particle/timestep/fov combinations
            mask = (
                (df_tracked["particle"].isin(table["particle"]))
                & (
                    df_tracked[timestep_name_variable]
                    == metadata[timestep_name_variable]
                )
                & (df_tracked["fov"] == metadata["fov"])
            )
            df_tracked.loc[mask, "ref_mean_intensity"] = df_tracked.loc[
                mask, "particle"
            ].map(table.set_index("particle")["ref_mean_intensity"])

        return df_tracked


# Backwards-compat alias
OptoCheckFE = RefFE
