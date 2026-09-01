from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GPModel:
    """Compatibility class matching local_ai_suite.core.GPModel in Aximo."""

    model: Any
    y_mean: float
    y_scale: float


def _numeric_matrix(
    frame: pd.DataFrame, features: list[str], medians: pd.Series
) -> np.ndarray:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(medians).to_numpy(dtype=float)


def predict_gp(
    bundle: dict[str, Any], design: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    features = bundle["features"]
    matrix = _numeric_matrix(design, features, bundle["medians"])
    scaled = bundle["scaler"].transform(matrix)
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    for gp in bundle["models"]:
        mean_standardized, sd_standardized = gp.model.predict(scaled, return_std=True)
        means.append(mean_standardized * gp.y_scale + gp.y_mean)
        standard_deviations.append(sd_standardized * gp.y_scale)
    return np.column_stack(means), np.column_stack(standard_deviations)

