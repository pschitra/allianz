# Allianz CVM Streamlit Dashboard — Fixed Deployment Package

## Important deployment note

Use the `app.py` in this package. It is fully self-contained and **does not require `src.analytics`, `src.data_loader`, or `src.modeling` imports**. This removes the `ModuleNotFoundError: No module named 'src'` issue shown on Streamlit Cloud.

Upload the extracted repository contents, preserving the `data/` folder. Do not upload only the ZIP file to GitHub. In Streamlit Community Cloud, set the main file path to `app.py`.

The app can also find the three Excel files if they are placed directly beside `app.py`, though the recommended structure is the included `data/` folder.

---

# Allianz CVM Intelligence Dashboard

An interactive Streamlit dashboard for the Ivey case **W27305: Allianz — Optimizing Customer Acquisition Strategy Using Machine Learning**. It combines the funnel, policy and regional data sets to diagnose T&B's conversion gap, profile customer value, explore geo-demographic opportunity and demonstrate clustering and classification.

## What is included

- Executive overview with channel benchmarking and an 18% conversion scenario
- Interactive funnel filters, leakage analysis, age/premium/cover signals and brand conversion
- Policy-book and customer-value analysis
- K-means policyholder segmentation with PCA visualization
- Regional opportunity scoring and life-stage/income analysis
- Logistic regression, random forest and gradient boosting classification
- Threshold tuning, ROC/AUC, confusion matrix, feature importance and campaign-capacity simulation
- Data explorer, missingness audit and CSV export
- An executed Jupyter notebook in `notebooks/Allianz_CVM_Analysis.ipynb`

## Repository structure

```text
allianz_cvm_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── NOTICE.md
├── smoke_test.py
├── run_dashboard.bat
├── run_dashboard.sh
├── .gitignore
├── .github/workflows/smoke-test.yml
├── .streamlit/
│   └── config.toml
├── data/
│   ├── W27307-XLS-ENG.xlsx
│   ├── W27308-XLS-ENG.xlsx
│   └── W27309-XLS-ENG.xlsx
├── notebooks/
│   └── Allianz_CVM_Analysis.ipynb
└── src/
    ├── __init__.py
    ├── analytics.py
    ├── data_loader.py
    └── modeling.py
```

## Run locally

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install and test:

```bash
pip install -r requirements.txt
python smoke_test.py
```

Run the dashboard:

```bash
streamlit run app.py
```

Windows users can also double-click `run_dashboard.bat`; macOS/Linux users can run `./run_dashboard.sh`.

The app opens at `http://localhost:8501`.

## Upload to GitHub

1. Create a new **private** GitHub repository.
2. Upload every file and folder in this project, preserving the folder structure.
3. Confirm that the three Excel files are inside `data/` and `config.toml` is inside `.streamlit/`.
4. Commit the files to the main branch.
5. GitHub Actions automatically runs the compile and smoke tests after every push.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app** and select your repository.
3. Set the main file path to `app.py`.
4. Deploy. Streamlit reads `requirements.txt` automatically.

No secrets are required.

## Data and modelling notes

- The app treats `status_report == "Policycreated"` as the conversion target.
- Final journey status is not included as a model feature, preventing direct target leakage.
- Regional fields are postcode-level characteristics and should not be interpreted as individual-level facts.
- The regional opportunity score is an exploratory weighted index, not a trained propensity score.
- The machine-learning results are educational demonstrations. A production implementation would require calibration, fairness testing, privacy review, drift monitoring and controlled experiments.

## Troubleshooting

**FileNotFoundError**  
Keep the Excel files in `data/` with their original names. Do not move them beside `app.py` unless you also update `src/data_loader.py`.

**Excel engine error**  
Run `pip install -r requirements.txt`; `openpyxl` is included.

**Deployment is slow on the first load**  
The first run reads three Excel files and caches the prepared data. Later page changes are faster.

**Machine-learning page retrains after changing settings**  
This is expected. Model results are cached for each model/test-size/random-seed combination.

## Classroom data notice

Read `NOTICE.md` before publishing the repository. The supplied case data may be restricted to authorised educational use.
