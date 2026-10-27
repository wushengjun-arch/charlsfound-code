# CHARLS-found

Core code for model training and cohort validation in CHARLS, HRS, KLoSA, ELSA and SHARE.
The repository includes data preparation, tokenization, encoder pretraining, tree baselines,
embedding probes, late fusion, bootstrap uncertainty and calibration checks.

## Structure

```text
charls_found/
  configs/               Data, model and cohort configuration
  data_audit/            Input schema and coding checks
  outcome_registry/      Outcome, timing and leakage preparation
  tokenization/          Variable maps, vocabulary and tokenized inputs
  supervised/            Supervised matrices and baseline models
  mlm/                   Encoder training, embeddings and CHARLS comparisons
  external_validation/   Cohort preparation, models and validation
docs/RUNNING.md           Execution order and commands
pyproject.toml            Package metadata and dependencies
requirements.txt         Editable installation shortcut
```

## Installation

Use Python 3.11 or later in a virtual environment, from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python charls_found/mlm/train_charls_found_lite.py --help
```

The pinned direct dependencies record the local packaging environment; they are not a
complete historical environment lock. PyTorch must support the intended compute platform.

## Data and execution

See [the running guide](docs/RUNNING.md) for the ordered workflow. Each command accepts
`--help`. Run commands from the repository root. Generated artifacts go under `outputs/`.

Data preparation expects the existing local MedHelp-compatible data service: the CHARLS
endpoint is set in `charls_found/configs/data.yaml`; cohort endpoints are in the cohort utility
modules. The service implementation, participant data and fitted checkpoints are not included.
Use authorized real cohort data and its original labels/codebooks. Existing prepared inputs
can be used at the paths in the configuration files; missing inputs must be supplied before
running their dependent steps.

The original study settings are retained, including the global continuous-bin setting in
`tokenization.yaml`. Repository cleanup does not replace the tokenizer or refit any model.
This core package excludes manuscript assembly, revision utilities, publication figure
generation and additional revision-specific analyses.

## Code checks

```bash
python -m pip install -e '.[dev]'
python -m black --check charls_found
python -m compileall -q charls_found
```

These checks validate source formatting and syntax, not model performance or scientific
reproducibility. The code is released under the MIT License (see [LICENSE](LICENSE)).

Repository: https://github.com/wushengjun-arch/charlsfound-code
Archived release DOI: to be added after the Zenodo release is published.
