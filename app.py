from __future__ import annotations

import html
import io
import os
import re
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from local_ai_suite.core import predict_gp


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("GP_MODEL_PATH", ROOT / "models" / "gp_model_bundle.joblib")
)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
DOSE_NORMALIZED_TARGET = "log10_Dose_normalized_conc"

app = FastAPI(
    title="GP Human PK Predictor",
    description="Prediction-only endpoint for a trusted Aximo Gaussian-process bundle.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Add models/gp_model_bundle.joblib to the repository before deployment."
        )

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict):
        raise ValueError("The model bundle is not a dictionary.")

    required = {"models", "scaler", "medians", "features", "targets"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"The model bundle is missing: {', '.join(missing)}")
    if not bundle["features"] or not bundle["targets"]:
        raise ValueError("The model bundle has no feature or target names.")
    if len(bundle["models"]) != len(bundle["targets"]):
        raise ValueError("The number of GP models and targets does not match.")
    return bundle


def _check_access_token(submitted: str) -> None:
    expected = os.environ.get("APP_ACCESS_TOKEN", "").strip()
    if expected and not secrets.compare_digest(submitted, expected):
        raise HTTPException(status_code=401, detail="Incorrect access token.")


def _dose_values(frame: pd.DataFrame, default_dose_mg: float) -> np.ndarray:
    if "Dose_mg" in frame.columns:
        dose = pd.to_numeric(frame["Dose_mg"], errors="coerce").to_numpy(float)
    elif "log10_Dose_mg" in frame.columns:
        log_dose = pd.to_numeric(
            frame["log10_Dose_mg"], errors="coerce"
        ).to_numpy(float)
        dose = np.power(10.0, log_dose)
    else:
        dose = np.full(len(frame), float(default_dose_mg), dtype=float)

    if not np.all(np.isfinite(dose)) or np.any(dose <= 0):
        raise HTTPException(
            status_code=422,
            detail="Dose must be positive and numeric. Supply Dose_mg, "
            "log10_Dose_mg, or a positive default dose.",
        )
    return dose


def _pow10(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.power(10.0, values)
    if not np.all(np.isfinite(transformed)):
        raise HTTPException(
            status_code=422,
            detail="Back-transformation produced a non-finite value. Check the inputs.",
        )
    return transformed


def _add_comparison_columns(result: pd.DataFrame, predicted: np.ndarray) -> None:
    actual = pd.to_numeric(
        result["Actual_concentration_ng_mL"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0) & (predicted > 0)

    signed_error = np.full(len(result), np.nan)
    absolute_error = np.full(len(result), np.nan)
    fold_error = np.full(len(result), np.nan)
    signed_error[valid] = 100.0 * (predicted[valid] - actual[valid]) / actual[valid]
    absolute_error[valid] = np.abs(signed_error[valid])
    ratio = predicted[valid] / actual[valid]
    fold_error[valid] = np.maximum(ratio, 1.0 / ratio)

    result["GP_signed_error_pct"] = signed_error
    result["GP_absolute_error_pct"] = absolute_error
    result["GP_fold_error"] = fold_error
    result["GP_within_1_5_fold"] = np.where(valid, fold_error <= 1.5, False)
    result["GP_within_2_fold"] = np.where(valid, fold_error <= 2.0, False)


def make_predictions(
    frame: pd.DataFrame, bundle: dict[str, Any], default_dose_mg: float
) -> pd.DataFrame:
    if frame.empty:
        raise HTTPException(status_code=422, detail="The uploaded CSV has no rows.")

    features = [str(value) for value in bundle["features"]]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail="Missing model feature columns: " + ", ".join(missing),
        )

    means, standard_deviations = predict_gp(bundle, frame)
    if means.shape != standard_deviations.shape or means.shape[0] != len(frame):
        raise HTTPException(status_code=500, detail="Unexpected GP prediction shape.")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(standard_deviations)):
        raise HTTPException(status_code=422, detail="The GP returned non-finite predictions.")

    result = frame.copy()
    targets = [str(value) for value in bundle["targets"]]
    for column_index, target in enumerate(targets):
        safe_target = re.sub(r"[^A-Za-z0-9_]+", "_", target).strip("_") or "target"
        mean = means[:, column_index]
        sd = np.maximum(standard_deviations[:, column_index], 0.0)
        result[f"GP_predicted_{safe_target}"] = mean
        result[f"GP_SD_{safe_target}"] = sd

        if target.casefold() == DOSE_NORMALIZED_TARGET.casefold():
            dose = _dose_values(frame, default_dose_mg)
            predicted = _pow10(mean) * dose
            lower = _pow10(mean - sd) * dose
            upper = _pow10(mean + sd) * dose
            result["GP_predicted_concentration_ng_mL"] = predicted
            result["GP_lower_1SD_concentration_ng_mL"] = lower
            result["GP_upper_1SD_concentration_ng_mL"] = upper

            if "Actual_concentration_ng_mL" in result.columns:
                _add_comparison_columns(result, predicted)

    return result


def _model_summary() -> tuple[str, str, str]:
    try:
        bundle = load_bundle()
        features = ", ".join(html.escape(str(value)) for value in bundle["features"])
        targets = ", ".join(html.escape(str(value)) for value in bundle["targets"])
        return "Ready", features, targets
    except Exception:
        return "Model file needed", "Unavailable until the model is added", "Unavailable"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    status, features, targets = _model_summary()
    protected = bool(os.environ.get("APP_ACCESS_TOKEN", "").strip())
    token_field = ""
    if protected:
        token_field = """
        <label for="access_token">Access token</label>
        <input id="access_token" name="access_token" type="password"
               autocomplete="current-password" required>
        """
    else:
        token_field = '<input name="access_token" type="hidden" value="">'

    escaped_status = html.escape(status)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#164e63">
  <title>GP Human PK Predictor</title>
  <style>
    :root {{ color-scheme: light; --ink:#17313a; --muted:#60737a; --paper:#f4f7f5;
      --card:#ffffff; --accent:#0f766e; --accent-dark:#115e59; --line:#d9e3df; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,ui-sans-serif,
      system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
    main {{ width:min(680px,100%); margin:0 auto; padding:32px 18px 56px; }}
    .eyebrow {{ color:var(--accent); font-size:.78rem; font-weight:750; letter-spacing:.1em;
      text-transform:uppercase; }}
    h1 {{ margin:.35rem 0 .65rem; font-size:clamp(1.9rem,8vw,3.15rem); line-height:1.02;
      letter-spacing:-.045em; }}
    .intro {{ color:var(--muted); margin:0 0 24px; max-width:58ch; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:18px;
      box-shadow:0 14px 38px rgba(20,55,60,.08); padding:20px; }}
    .status {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
      padding-bottom:16px; margin-bottom:18px; border-bottom:1px solid var(--line); }}
    .badge {{ border-radius:999px; padding:6px 10px; color:#0b4f47; background:#d9f1eb;
      font-size:.8rem; font-weight:700; }}
    label {{ display:block; margin:16px 0 7px; font-size:.9rem; font-weight:700; }}
    input {{ width:100%; min-height:48px; border:1px solid #bdccc7; border-radius:10px;
      background:#fff; color:var(--ink); padding:11px 12px; font:inherit; }}
    input:focus {{ outline:3px solid rgba(15,118,110,.18); border-color:var(--accent); }}
    input[type=file] {{ padding:9px; }}
    button {{ width:100%; min-height:50px; margin-top:20px; border:0; border-radius:11px;
      background:var(--accent); color:white; font:inherit; font-weight:800; cursor:pointer; }}
    button:hover {{ background:var(--accent-dark); }} button:disabled {{ opacity:.6; cursor:wait; }}
    .message {{ display:none; margin:14px 0 0; border-radius:9px; padding:10px 12px;
      background:#fff0ed; color:#8a3328; font-size:.88rem; }}
    details {{ margin-top:18px; border-top:1px solid var(--line); padding-top:14px; }}
    summary {{ cursor:pointer; font-weight:700; }}
    .meta {{ color:var(--muted); font-size:.84rem; overflow-wrap:anywhere; }}
    .privacy {{ margin:16px 2px 0; color:var(--muted); font-size:.82rem; }}
    code {{ background:#eef3f1; border-radius:5px; padding:2px 5px; }}
    @media (max-width:420px) {{ main {{ padding-top:22px; }} .card {{ padding:17px; }} }}
    @media (prefers-reduced-motion:reduce) {{ * {{ scroll-behavior:auto!important; }} }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Prediction only</div>
    <h1>GP Human PK Predictor</h1>
    <p class="intro">Upload human PK conditions, run the bundled Gaussian-process model,
      and download the predictions as a CSV. No training is performed here.</p>
    <section class="card">
      <div class="status"><strong>Service status</strong><span class="badge">{escaped_status}</span></div>
      <form action="/predict" method="post" enctype="multipart/form-data" id="form">
        <label for="file">Human PK data (.csv)</label>
        <input id="file" name="file" type="file" accept=".csv,text/csv" required>
        <label for="default_dose_mg">Default dose (mg)</label>
        <input id="default_dose_mg" name="default_dose_mg" type="number" value="0.25"
               min="0.000000001" step="any" inputmode="decimal" required>
        {token_field}
        <button id="submit" type="submit">Calculate and download CSV</button>
        <p class="message" id="message" role="status" aria-live="polite"></p>
      </form>
      <details>
        <summary>Model input details</summary>
        <p class="meta"><strong>Required columns:</strong> {features}</p>
        <p class="meta"><strong>Model output:</strong> {targets}</p>
        <p class="meta">If <code>Dose_mg</code> is absent, the app uses
          <code>log10_Dose_mg</code>, then the default dose above.</p>
      </details>
    </section>
    <p class="privacy">Files are processed in memory and are not deliberately stored.
      Protect this deployment before using confidential data.</p>
  </main>
  <script>
    const form = document.getElementById('form');
    const button = document.getElementById('submit');
    const message = document.getElementById('message');
    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      button.disabled = true;
      button.textContent = 'Calculating…';
      message.style.display = 'none';
      try {{
        const response = await fetch('/predict', {{ method: 'POST', body: new FormData(form) }});
        if (!response.ok) {{
          let detail = 'Prediction failed.';
          try {{ detail = (await response.json()).detail || detail; }} catch (_) {{}}
          throw new Error(detail);
        }}
        const blob = await response.blob();
        const disposition = response.headers.get('content-disposition') || '';
        const match = disposition.match(/filename="?([^";]+)"?/i);
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = match ? match[1] : 'human_pk_gp_predictions.csv';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(link.href);
      }} catch (error) {{
        message.textContent = error.message;
        message.style.display = 'block';
      }} finally {{
        button.disabled = false;
        button.textContent = 'Calculate and download CSV';
      }}
    }});
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        bundle = load_bundle()
        return {
            "status": "ready",
            "feature_count": len(bundle["features"]),
            "features": [str(value) for value in bundle["features"]],
            "targets": [str(value) for value in bundle["targets"]],
        }
    except Exception as exc:
        return {"status": "model_not_ready", "message": str(exc)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    default_dose_mg: float = Form(0.25),
    access_token: str = Form(""),
) -> StreamingResponse:
    _check_access_token(access_token)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    if not np.isfinite(default_dose_mg) or default_dose_mg <= 0:
        raise HTTPException(status_code=422, detail="Default dose must be positive.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV files are limited to 2 MB.")

    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {exc}") from exc

    try:
        bundle = load_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"GP model is unavailable: {exc}") from exc

    result = make_predictions(frame, bundle, default_dose_mg)
    output = io.BytesIO(result.to_csv(index=False).encode("utf-8-sig"))
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file.filename).stem).strip("_")
    stem = stem[:80] or "human_pk"
    headers = {
        "Content-Disposition": f'attachment; filename="{stem}_gp_predictions.csv"',
        "Cache-Control": "no-store, max-age=0",
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(output, media_type="text/csv; charset=utf-8", headers=headers)
