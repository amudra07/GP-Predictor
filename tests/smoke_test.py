from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_ai_suite.core import GPModel  # noqa: E402


def build_temporary_bundle(model_path: Path) -> None:
    features = ["log1p_Time_h", "log10_BW_kg", "log10_Dose_mg"]
    matrix = np.array(
        [
            [0.69, 0.70, -1.30],
            [1.10, 0.70, -1.30],
            [1.61, 0.70, -1.30],
            [2.20, 1.00, -0.90],
            [2.83, 1.00, -0.90],
            [3.43, 1.30, -0.60],
            [3.89, 1.30, -0.60],
            [4.39, 1.60, -0.30],
            [4.80, 1.60, -0.30],
            [5.20, 1.90, 0.00],
        ],
        dtype=float,
    )
    target = 3.15 - (0.32 * matrix[:, 0]) + (0.12 * matrix[:, 1])
    scaler = StandardScaler().fit(matrix)
    standardized_target = (target - target.mean()) / target.std()
    kernel = ConstantKernel(1.0) * RBF(np.ones(3)) + WhiteKernel(1e-4)
    model = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(
        scaler.transform(matrix), standardized_target
    )
    bundle = {
        "kind": "gaussian_process_bo",
        "models": [GPModel(model=model, y_mean=target.mean(), y_scale=target.std())],
        "scaler": scaler,
        "medians": pd.Series(np.median(matrix, axis=0), index=features),
        "features": features,
        "targets": ["log10_Dose_normalized_conc"],
    }
    joblib.dump(bundle, model_path)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "gp_model_bundle.joblib"
        build_temporary_bundle(model_path)
        os.environ["GP_MODEL_PATH"] = str(model_path)

        import app as predictor_app

        client = TestClient(predictor_app.app)
        sample = (ROOT / "sample_data" / "human_pk_input_example.csv").read_bytes()
        response = client.post(
            "/predict",
            files={"file": ("human.csv", sample, "text/csv")},
            data={"default_dose_mg": "0.25", "access_token": ""},
        )
        assert response.status_code == 200, response.text
        assert "attachment" in response.headers["content-disposition"]

        output = pd.read_csv(io.BytesIO(response.content))
        log_prediction = output["GP_predicted_log10_Dose_normalized_conc"]
        expected = np.power(10.0, log_prediction) * output["Dose_mg"]
        np.testing.assert_allclose(
            output["GP_predicted_concentration_ng_mL"], expected, rtol=1e-12
        )
        assert output["GP_fold_error"].notna().all()
        assert client.get("/health").json()["status"] == "ready"
        assert "GP Human PK Predictor" in client.get("/").text
        print(f"Smoke test passed: {len(output)} rows predicted and downloaded.")


if __name__ == "__main__":
    main()

