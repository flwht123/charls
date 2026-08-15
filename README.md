# CHARLS depression-trajectory analysis code

Analysis code accompanying the manuscript by Yuan H and Wang H on baseline risk
stratification of persistently high depressive-symptom trajectories in CHARLS.

## Data

No participant-level data are distributed here. CHARLS and Harmonized CHARLS remain subject to
their own data-use terms. The analysis starts from a local `final_analytic_dataset.csv`, whose
required model columns are listed in `reproduce_analysis.py`.

## Run

Python 3.12.10 with the package versions in `requirements.txt`; seed 42.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python reproduce_analysis.py --data final_analytic_dataset.csv
```

Outputs are written to `results/`. A full run takes several hours; `--quick` is a technical
smoke test only.

## Released boundary

Released: the analysis from `final_analytic_dataset.csv` onward, covering preprocessing inside
the modelling pipeline, model fitting, recalibration, decision curve analysis, SHAP, the
downstream sensitivity and robustness analyses, and Figures 3 to 6.

Not released: the data-preparation code that builds `final_analytic_dataset.csv` from CHARLS,
Figure 1, the trajectory plot in Figure 2, and the upstream cohort-construction and descriptive
tables.

`figure_source_values/` is the released copy of the aggregate values behind the modelling
results: for the submitted figures, the ROC coordinates and pointwise 95% bootstrap-band limits
of Figure 3, the confusion-matrix cells of Figure 4, the calibration points and net-benefit
values of both panels of Figure 5, and the per-feature mean absolute SHAP values with their
ranges of Figure 6; the remaining files hold the downstream modelling results. Nothing outside
that boundary is covered, no file contains participant-level rows, and all are the output of the
run recorded in `figure_source_values/run_manifest.json`.

## License

The MIT license applies only to the source code in this repository, not to CHARLS data or any
participant-level derivative.
