# GP Human PK Predictor

A small, prediction-only FastAPI application for Vercel. It loads one trusted Gaussian-process bundle exported by Aximo, accepts a human PK CSV, and downloads a CSV containing model predictions and uncertainty.

It does **not** train or modify the GP model.

## 1. Add your GP model

In Aximo, download the trained Gaussian-process model bundle. Rename it if needed and put it here:

```text
models/gp_model_bundle.joblib
```

The application contains the minimal `local_ai_suite.core.GPModel` compatibility class needed to load the Aximo Joblib bundle.

> Security: never load a Joblib file from an unknown source. Joblib uses pickle and a malicious file can execute code. This app intentionally does not provide a model-upload endpoint.

## 2. Prepare the human CSV

The CSV must contain every feature listed in the trained bundle. Extra columns are preserved in the downloaded result.

For the GLP-1 comparison model, the typical required transformed fields are:

```text
log1p_Time_h
log10_BW_kg
log10_Dose_mg
```

Use the exact feature names shown on the web page after the model is installed. See `sample_data/human_pk_input_example.csv` for an example.

If the target is `log10_Dose_normalized_conc`, the app automatically calculates:

```text
concentration_ng_mL = 10 ** predicted_log10_Dose_normalized_conc * Dose_mg
```

Dose is selected in this order:

1. The row's `Dose_mg` value.
2. `10 ** log10_Dose_mg`.
3. The default dose entered on the page.

When `Actual_concentration_ng_mL` is present, the output also contains signed error, absolute percentage error, symmetric fold error, and within-1.5-fold/within-2-fold flags.

## 3. Upload to GitHub

Create a **private** GitHub repository and upload the contents of this folder—not the outer folder itself. Confirm that these paths are visible at the repository root:

```text
app.py
requirements.txt
models/gp_model_bundle.joblib
local_ai_suite/core.py
```

GitHub blocks individual files larger than 100 MiB. If the model alone is near that size, Vercel is not the best packaging route.

## 4. Deploy on Vercel

1. In Vercel, choose **Add New → Project**.
2. Import the private GitHub repository.
3. Leave Framework Preset, Build Command, and Output Directory at their automatic/default values.
4. In **Environment Variables**, add `APP_ACCESS_TOKEN` with a long private value.
5. Deploy.

Open the deployment URL, enter the token, choose the human CSV, and select **Calculate and download CSV**.

The app exposes `/health` for a simple model readiness check.

## 5. Run locally (optional)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`.

## Output columns

The original input columns are retained. The app adds:

- `GP_predicted_<target>`: GP mean on the model's target scale.
- `GP_SD_<target>`: GP standard deviation on the target scale.
- `GP_predicted_concentration_ng_mL`: back-transformed point prediction for the GLP-1 target.
- `GP_lower_1SD_concentration_ng_mL` and `GP_upper_1SD_concentration_ng_mL`: back-transformed mean ± 1 GP standard deviation.
- Comparison columns when actual concentration is supplied.

The ±1 SD interval is model uncertainty, not a guaranteed clinical confidence interval. This is a comparison/research utility and not a dosing or clinical-decision system.

## Size and privacy notes

- The application code is tiny; most deployment size comes from NumPy, pandas, SciPy, and scikit-learn.
- Uploads are restricted to 2 MB, processed in memory, returned immediately, and not deliberately stored by this code.
- Vercel still operates the infrastructure, so do not upload identifiable or regulated clinical data without your institution's approval and an appropriate deployment/privacy setup.
- Keep the repository private because it contains your trained model.

