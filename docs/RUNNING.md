# Running the core workflow

Install the package in editable mode and run from the repository root. Prepare authorized
data and a compatible local data service before preprocessing. Training commands require
their upstream matrices, vocabularies and split manifests; validation additionally requires
trained checkpoints and exported embeddings. No inputs or results are fabricated.

## CHARLS

### Input checks and outcome preparation

```bash
python charls_found/data_audit/audit_charls_schema.py
```

```bash
python charls_found/outcome_registry/build_candidate_outcomes.py
python charls_found/outcome_registry/validate_outcomes.py
python charls_found/outcome_registry/build_time_alignment.py
python charls_found/outcome_registry/build_leakage_exclusion_table.py
```

### Tokenization and supervised inputs

```bash
python charls_found/tokenization/build_variable_type_map.py
python charls_found/tokenization/build_value_label_map.py
python charls_found/tokenization/build_bins.py
python charls_found/tokenization/build_vocabularies.py
python charls_found/tokenization/tokenize_person_wave.py
python charls_found/tokenization/tokenization_qc.py
```

```bash
python charls_found/supervised/build_supervised_outcome_matrix.py
python charls_found/supervised/train_baseline_probe.py
```

### Encoder training and embedding export

```bash
python charls_found/mlm/prepare_mlm_inputs.py
python charls_found/mlm/train_charls_found_lite.py
python charls_found/mlm/export_person_wave_embeddings.py
```

Export the diagnostic-masked embeddings required by the configured comparison before running it:

```bash
python charls_found/mlm/export_person_wave_embeddings.py --mask-mode diagnostic --outcome-ids incident_diabetes incident_hypertension multimorbidity_onset
python charls_found/mlm/compare_embedding_probe.py
python charls_found/mlm/late_fusion_probe.py
```

The optional `multimorbidity_stability.py` and `late_fusion_stability.py` commands in `mlm/`
retain the original bootstrap and repeated-split analyses.

## Other cohorts

Each cohort has its own preprocessing and training implementation. In the commands below,
replace `hrs` with `klosa`, `elsa` or `share` for the corresponding cohort. Keep this order.

```bash
python charls_found/external_validation/hrs_build_crosswalk.py
python charls_found/external_validation/hrs_standardize_long.py
python charls_found/external_validation/hrs_build_leakage_table.py
python charls_found/external_validation/hrs_build_supervised_matrix.py
python charls_found/external_validation/hrs_train_baselines.py
python charls_found/external_validation/hrs_prepare_mlm_inputs.py
python charls_found/external_validation/hrs_train_encoder.py
python charls_found/external_validation/hrs_export_embeddings.py
python charls_found/external_validation/hrs_embedding_vs_tree.py
```

The cohort crosswalk commands also use the prepared CHARLS variable map. KLoSA, ELSA and
SHARE wrappers use `build_crosswalks.py`; this shared dependency is included. Cohort
embedding exporters produce both unmasked and the configured diagnostic-masked embeddings.

`<cohort>_multimorbidity_stability.py` provides the corresponding optional stability analysis.

## Fusion and validation

After the cohort's comparison step, run fusion and bootstrap validation. For another cohort,
replace both `hrs` arguments and the configuration filename consistently.

```bash
python charls_found/external_validation/late_fusion.py --cohort hrs --config charls_found/configs/hrs_mlm.yaml
python charls_found/external_validation/bootstrap_deltas.py --cohort hrs --config charls_found/configs/hrs_mlm.yaml
python charls_found/external_validation/fusion_calibration.py --cohorts hrs
```

For CHARLS bootstrap validation use `--cohort charls --config charls_found/configs/mlm.yaml`.
`fusion_calibration.py --cohorts charls hrs klosa elsa share` processes all five prepared cohorts.
Use each command's `--help` for bootstrap iterations, output paths and other supported options.

All default outputs stay inside `outputs/`. Existing artifact names and model settings are
preserved so upstream and downstream steps continue to agree. The calibration command's
additional figures and reports go to `outputs/figures/` and `outputs/reports/`.

## Scope of the prepared release

Source naming, formatting, package paths and documentation were cleaned for distribution.
The data definitions, category mappings, model parameters and statistical calculations were
not changed. Preparing this repository did not rerun data analysis or verify every study result.
