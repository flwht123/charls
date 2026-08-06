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

## Released boundary

This repository does not contain the complete analysis code for the study. It contains the
analysis code from `final_analytic_dataset.csv` onward, which covers preprocessing inside the
modelling pipeline, model fitting, recalibration, decision curve analysis, SHAP computation,
the sensitivity and robustness analyses, and the generation of every reported table and figure,
including the submitted compositions for Figures 3 to 6. The upstream data-preparation code
that builds `final_analytic_dataset.csv` from CHARLS is not released.

## Figure source values

Figures 3 to 6 of the submitted manuscript are the unmodified output of this repository. Running
`reproduce_analysis.py` writes `main_Fig3_roc.png`, `main_Fig4_confusion.png`,
`main_Fig5_calibration_dca.png` and `main_Fig6_shap.png`, which reproduce the submitted files
and are byte-identical to them on the locked environment recorded in `requirements.txt`; on
other platforms or font configurations the rendered bytes may differ while the plotted values
do not. Figure 1 is a study flow diagram drawn by hand and Figure 2 is the trajectory plot;
neither depends on the modelling code.

`figure_source_values/` holds the aggregate values behind the two composite figures, so they can
be checked without running the analysis and without access to participant-level data: the
calibration-curve points and net-benefit values for both panels of Figure 5, and the per-feature
mean absolute SHAP values with their ranges for Figure 6. These files contain no
participant-level rows, and they are the output of the run recorded in
`figure_source_values/run_manifest.json`. Running the script writes the aggregate values behind
every other reported figure and table to the output directory as well.


## Preprocessing note

Predictors are declared by measurement scale before modelling. Binary predictors enter as
indicator columns, ordinal predictors (educational attainment, self-rated health) keep their
ordinal codes, and the single nominal predictor (marital status) is one-hot encoded inside the
pipeline with married as the reference category. Imputation, scaling, and encoding are all
fitted on training folds only. In the multiple-imputation sensitivity analysis the imputation
model is fitted within the training set and then applied to both partitions.

## License

The MIT license applies only to the source code in this repository, not to CHARLS data or any
participant-level derivative.
