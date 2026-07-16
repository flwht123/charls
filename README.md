# CHARLS depression-trajectory analysis code

Analysis code accompanying the manuscript by Yuan H and Wang H on baseline risk
stratification of persistently high depressive-symptom trajectories in CHARLS.

## Data

The participant-level analytic dataset is not included. CHARLS and Harmonized CHARLS data are
subject to their respective data-use terms. This repository intentionally does not distribute
raw data, derived participant-level data, or the data-preparation code.

The analysis starts from a local file named `final_analytic_dataset.csv`. Its construction,
variable definitions, inclusion criteria, and preprocessing procedures are described in the
manuscript and supplementary material. The required model columns are listed explicitly in
`reproduce_analysis.py`.

## Run

The submitted analysis used Python 3.12.10 and the package versions in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python reproduce_analysis.py --data final_analytic_dataset.csv
```

Aggregate tables and figures are written to `results/`. The complete analysis can take several
hours. `--quick` reduces the number of trees, bootstrap resamples, and repeated splits and is
intended only as a technical smoke test.

Seed 42 is used for randomized analysis steps.

## License

The MIT license applies only to the source code in this repository, not to CHARLS data or any
participant-level derivative.
