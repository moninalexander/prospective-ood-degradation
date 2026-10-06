# prospective-ood-degradation

This repository contains the code accompanying our study of whether the onset of out-of-distribution degradation can be predicted prospectively from training-side information alone. We use training dynamics to predict an upcoming degradation event without using OOD performance as an input. The accompanying paper is available at [`paper/prospective_prediction.pdf`](paper/prospective_prediction.pdf).


## Dataset

We use a binary version of MNIST, with digits \(0\)-\(4\) assigned to class 0 and digits \(5\)-\(9\) to class 1. The grayscale images are converted to three channels.

The source dataset contains a \(1\times1\) color shortcut in the upper-left corner that is perfectly correlated with the class: red for class 0 and blue for class 1. The clean dataset contains the same binary MNIST task without the shortcut and is used to measure OOD degradation.

The shortcut dataset is implemented in [`definitions/data.py`](definitions/data.py). Dataset configurations, including the shortcut probabilities and patch size, are defined in [`configs/configs.py`](configs/configs.py).

## Models

The models are defined in [`definitions/models.py`](definitions/models.py), while the architectures used in the experiments are specified in [`configs/configs.py`](configs/configs.py).

The main CNN used in the paper has two convolutional layers,

```text
3 channels -> Conv(3 channels, 2x2) -> ReLU -> MaxPool(2x2)
           -> Conv(3 channels, 2x2) -> ReLU -> MaxPool(2x2)
           -> Linear(4) -> ReLU -> Linear(2).
```

We also use a small MLP for the architecture-transfer experiment,

```text
Flatten -> Linear(2352, 4) -> ReLU
        -> Linear(4, 2) -> ReLU
        -> Linear(2, 2).
```

The logistic-regression predictor used to predict future degradation is also defined in [`definitions/models.py`](definitions/models.py).

## Running the experiments

Before running the code, copy

```text
configs/local_config.EXAMPLE.py
```

to

```text
configs/local_config.py
```

and set `DATA_ROOT` to the location where the MNIST data and experimental results should be stored. In particular,

```python
EXPERIMENTS_DIR = DATA_ROOT / "prospective-ood-degradation"
```

specifies the root directory for the generated experimental data. The local configuration file is excluded from Git.

### 1. Generate training trajectories

Run

```bash
python experiments/train_models.py
```

to train models and collect the training-side metrics used in the analysis.

The script currently uses only two random seeds for each shortcut-trained architecture as a small example, together with one clean-training trajectory. The paper used substantially larger cohorts.

Each trajectory is stored under

```text
EXPERIMENTS_DIR/
    <dataset>/
        <architecture>/
            <run_name>/
                seed_<seed>/
```

and contains

```text
config.json
metrics.pt
model.pt
```

The recorded quantities include training loss and accuracy, gradient and parameter norms, parameter changes, confidence, and entropy, together with source and clean evaluation quantities used offline to define the degradation event.

### 2. Construct the prediction datasets

Run

```bash
python experiments/build_prediction_dataset.py
```

to identify eligible trajectories, split them into predictor training and held-out sets, and construct window features from the recorded training metrics.

For every metric, the code computes four features over a sliding window:

```text
current value
mean
standard deviation
slope
```

The generated files are saved automatically inside the corresponding experimental run directory. Their filenames encode the analysis parameters. With the settings used in the paper, the prediction dataset is

```text
PREDICTION_DT0.05_MD0.05_PW90_PR81_PDF0_WF4_PDS500_WS100_PH200_SS7.pt
```

and the corresponding full-history dataset is

```text
FULLHIST_DT0.05_MD0.05_PW90_PR81_PDF0_WF4_PDS500_WS100_PH200.pt
```

Here, for example, `WS100` denotes a window size of 100 training steps and `PH200` a prediction horizon of 200 steps.

### 3. Train and evaluate the predictors

Run

```bash
python experiments/predictor_logistic_regression.py
```

to train the logistic-regression predictor on the window features and evaluate it on the held-out CNN trajectories and on the MLP trajectories.

The predictor uses the following nine training-side quantities:

```text
train_loss
train_accuracy
grad_norm
param_norm
delta_local_param_norm
delta_global_param_norm
confidence
confidence_correct
entropy
```

with the four window features described above.

The time-only baseline can be run with

```bash
python experiments/predictor_time_only.py
```

This uses the absolute training step as its only input and otherwise follows the same train/test trajectory split and prediction protocol.

## Additional visualization scripts

[`experiments/plot_trajectories.py`](experiments/plot_trajectories.py) plots the clean and source evaluation-accuracy trajectories produced during training. It is useful for inspecting the development of OOD degradation across individual runs.

[`experiments/visualize_MNISTShortcut.py`](experiments/visualize_MNISTShortcut.py) displays examples of the binary MNIST images with the red/blue shortcut added to the upper-left corner.

## Requirements

The code uses Python with PyTorch and the following packages:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `rich`

Scripts should be run from the repository root.