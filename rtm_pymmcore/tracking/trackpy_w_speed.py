import numpy as np
import pandas as pd
from rtm_pymmcore.tracking.trackpy import TrackerTrackpy


class TrackerTrackpyWithSpeed(TrackerTrackpy):
    """
    Tracker that extends TrackerTrackpy to calculate speed, displacement, and area changes.

    This tracker calculates:
    - speed: instantaneous speed between consecutive frames (pixels/frame)
    - displacement: cumulative displacement from the first appearance of each particle
    - area_change: change in area between consecutive frames
    """

    def _post_process_tracking(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Override to add speed, displacement, and area change calculations.

        Args:
            df: Tracked dataframe from parent class
            metadata: Metadata dictionary

        Returns:
            pd.DataFrame: Dataframe with added motion metrics
        """
        return self._calculate_motion_metrics(df)

    def _calculate_motion_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate speed, displacement, and area changes for tracked particles.
        Only calculates metrics for the most recent frame to avoid redundant calculations.

        Args:
            df: Dataframe with tracked particles

        Returns:
            pd.DataFrame: Dataframe with added motion metric columns
        """
        # Initialize columns if they don't exist
        if "speed" not in df.columns:
            df["speed"] = np.nan
        if "displacement" not in df.columns:
            df["displacement"] = 0.0
        if "area_change" not in df.columns:
            df["area_change"] = np.nan

        # Get the current (most recent) fov_timestep
        current_timestep = df["fov_timestep"].max()

        # Only process particles that appear in the current timestep
        current_particles = df[df["fov_timestep"] == current_timestep][
            "particle"
        ].unique()

        for particle_id in current_particles:
            particle_mask = df["particle"] == particle_id
            particle_data = df[particle_mask].sort_values("fov_timestep")

            if len(particle_data) < 1:
                continue

            # Get the index of the current (last) entry for this particle
            current_idx = particle_data.index[-1]

            if len(particle_data) == 1:
                # First appearance of this particle
                df.loc[current_idx, "speed"] = 0.0
                df.loc[current_idx, "displacement"] = 0.0
                df.loc[current_idx, "area_change"] = 0.0
            else:
                # Not the first frame - calculate metrics relative to previous frame
                previous_idx = particle_data.index[-2]

                # Calculate speed (distance from previous position)
                dx = df.loc[current_idx, "x"] - df.loc[previous_idx, "x"]
                dy = df.loc[current_idx, "y"] - df.loc[previous_idx, "y"]
                speed = np.sqrt(dx**2 + dy**2)
                df.loc[current_idx, "speed"] = speed

                # Calculate cumulative displacement from first position
                first_idx = particle_data.index[0]
                dx_total = df.loc[current_idx, "x"] - df.loc[first_idx, "x"]
                dy_total = df.loc[current_idx, "y"] - df.loc[first_idx, "y"]
                displacement = np.sqrt(dx_total**2 + dy_total**2)
                df.loc[current_idx, "displacement"] = displacement

                # Calculate area change from previous frame
                if "area" in df.columns:
                    area_change = (
                        df.loc[current_idx, "area"] - df.loc[previous_idx, "area"]
                    )
                    df.loc[current_idx, "area_change"] = area_change
                else:
                    df.loc[current_idx, "area_change"] = 0.0

        return df
