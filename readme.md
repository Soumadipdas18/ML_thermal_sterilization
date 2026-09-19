# Physics-Regularized Recurrent Neural Network for Thermal Sterilization

This repository contains the machine-learning workflow developed for rapid surrogate modeling of the thermal sterilization of canned solid-liquid foods. The model is a **physics-regularized recurrent neural network (PRNN)** that predicts the temperature history at the slowest heating zone (SHZ), thermal lethality, and a quality-retention metric from the operating conditions of the thermal process.

The framework was developed to complement computational fluid dynamics (CFD) simulations. Once trained, the surrogate can evaluate operating conditions much faster than repeated CFD simulations while retaining a physics-based consistency relation between the predicted temperature trajectory and the corresponding lethality.

## Model overview

For each thermal-processing condition, the model receives four operating inputs:

- come-up time (CUT),
- constant-temperature heating time,
- cooling time, and
- heating-stage retort temperature.

The model predicts:

1. the SHZ temperature trajectory during heating and cooling,
2. the process lethality expressed as an F-value, and
3. a model-based quality-retention index.

The temperature trajectory is generated using a recurrent sequence decoder. Separate regression heads predict the scalar lethality and quality-retention outputs.

A physics-regularization term links the independently predicted F-value to the F-value obtained by integrating the lethal rate along the predicted temperature trajectory. The physical relation is therefore imposed as a **soft penalty**, rather than as an exact hard constraint. For this reason, the model is described as a physics-regularized neural network rather than a strictly constrained physics-informed neural network.

## Workflow

```text
Operating conditions
        |
        v
Input normalization / encoder
        |
        +-----------------------------+
        |                             |
        v                             v
LSTM sequence decoder          Scalar regression heads
        |                             |
        v                             v
SHZ temperature trajectory      F-value + quality retention
        |
        v
Lethality integration
        |
        v
Trajectory-derived F-value
        |
        v
Physics-consistency loss
```

## Dataset

The workflow uses a CFD-generated dataset containing **10,526 thermal-processing simulations** for canned peas in water.

The operating variables represented in the dataset include:

| Variable | Description |
|---|---|
| `CUT` | Retort come-up time |
| `time_heat` | Constant-temperature heating duration |
| `time_cool` | Cooling duration |
| `temp_retort` | Heating-stage retort temperature |

The primary targets include:

| Target | Description |
|---|---|
| `temp_heat_arr` | SHZ temperature history during heating |
| `temp_cool_arr` | SHZ temperature history during cooling |
| `F_SHZ` | Lethality calculated at the SHZ |
| `quality` | Model-based quality-retention index |

The current study uses a target lethality of **F0 = 3 min at 121.1 °C**, with the configured threshold defined in the model scripts.

## Repository workflow

The analysis is performed in two stages.

### 1. Model selection and ablation analysis

Run:

```bash
python 01_prnn_ablation_selection.py
```

This script:

- loads and audits `Dataset_all.csv`,
- creates reproducible training, validation, and untouched test subsets,
- fits normalization parameters using the training subset only,
- selects the temperature-trajectory resolution,
- evaluates model capacity and dropout,
- selects the physics-regularization weight,
- compares the PRNN-LSTM with baseline architectures, and
- saves the selected configuration, frozen data split, normalization parameters, and validation results required for the final evaluation.

The untouched test set is not used during model or hyperparameter selection.

### 2. Final training and evaluation

After completing the first script, run:

```bash
python 02_final_prnn_lstm_evaluation.py
```

This script:

- loads the configuration selected using validation data,
- reuses the frozen data split,
- trains and evaluates the selected architecture using predefined random seeds,
- evaluates the untouched test set,
- performs structured holdout tests,
- evaluates trajectory, lethality, quality-retention, and physics-consistency errors,
- exports prediction tables and summary metrics, and
- identifies candidate operating conditions for subsequent CFD verification.

No architecture or hyperparameter is modified in response to test-set performance.

## Model comparison

The model-selection workflow compares the physics-regularized LSTM against relevant data-driven alternatives, including:

- the same recurrent model without the physics penalty,
- a GRU-based sequence model, and
- a time-conditioned multilayer perceptron.

The comparison uses the same data split and a common validation score so that model selection is performed consistently.

## Physics regularization

The thermal lethality associated with a predicted temperature history is evaluated from the lethal-rate integral. In general form,

[
F_0 = \int 10^{\frac{T(t)-T_{\mathrm{ref}}}{z}}\,dt,
]

where the reference temperature used in this work is

[
T_{\mathrm{ref}} = 121.1~^{\circ}\mathrm{C}.
]

The trajectory-derived F-value is compared with the F-value predicted directly by the scalar output head. Their disagreement contributes to the training loss and encourages physically consistent predictions.

## Evaluation

The workflow reports metrics for several aspects of model performance, including:

- temperature-trajectory RMSE and MAE,
- direct F-value prediction error,
- trajectory-derived F-value error,
- agreement between direct and trajectory-derived F-values,
- quality-retention RMSE, MAE, and R²,
- heating-to-cooling trajectory continuity,
- behavior near the target lethality threshold,
- inference time, and
- structured holdout performance.

Structured holdout tests are included to distinguish performance under interpolation from performance when the model is evaluated farther from the training data.

## Installation

A recent Python 3 environment is recommended. The main dependencies are:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib xlsxwriter
```

The scripts were designed to run conveniently on Kaggle. GPU acceleration is recommended for the full model-selection workflow. When compatible GPUs are available, the code can use TensorFlow acceleration and mixed precision.

## Running on Kaggle

1. Upload `Dataset_all.csv` as a Kaggle dataset or place it in the working directory.
2. Add the repository scripts to the notebook environment.
3. Enable a GPU accelerator.
4. Run `01_prnn_ablation_selection.py` first.
5. Preserve the output directory generated by the first script.
6. Run `02_final_prnn_lstm_evaluation.py` using the saved selection outputs.

If `DATA_PATH` is left as `None`, the scripts search common Kaggle input locations for `Dataset_all.csv`.

## Main outputs

Depending on the workflow stage, the scripts export files such as:

```text
selected_config.json
normalization_training_only.json
data_splits.npz
software_hardware.json
trained_weights/
CSV/XLSX result tables
prediction tables
holdout results
candidate conditions for CFD verification
```

These files document the selected model, frozen data partition, normalization parameters, model predictions, evaluation metrics, and computational environment.

## Interpretation and limitations

This model is a **surrogate of the CFD framework represented by the training dataset**. It should therefore be interpreted within the sampled operating and physical domain.

In particular:

- extrapolation outside the represented operating domain is not guaranteed,
- the model does not replace independent CFD or experimental verification,
- candidate operating conditions identified by the surrogate should be verified independently before use, and
- the predicted conditions should not be interpreted as certified commercial sterilization schedules.

The quality output is a model-based retention index defined by the assumptions used in the underlying CFD/data-generation framework.

## Associated research

This repository supports the work on a physics-regularized recurrent neural-network surrogate for thermal sterilization of canned solid-liquid foods. The model combines CFD-generated process data, recurrent sequence learning, and a lethality-based physics-consistency term to enable rapid screening of thermal-processing conditions.

If you use this repository in academic work, please cite the associated manuscript once its final bibliographic information is available.

## Repository

[Soumadipdas18/ML_thermal_sterilization](https://github.com/Soumadipdas18/ML_thermal_sterilization)
