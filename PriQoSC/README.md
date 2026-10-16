# PriQoSC

PriQoSC provides privacy-preserving QoS-aware service composition through representative candidate-pool construction and differentially private selection. The project includes utility experiments, parameter ablations, and two preference inference attacks.

## 1. Environment Setup

Use Python 3.9 with the dependencies listed in `requirements.txt`. Run all commands below from the project root.

```bash
python -m pip install -r requirements.txt
```

Place the dataset files in the following directories:

```text
data/
├── QWS/
│   ├── nodefeatures.data
│   └── serviceFeature.data
└── Normal/
    ├── nodefeatures.data
    └── serviceFeature.data
```

Replace `<dataset>` with `QWS` or `Normal` in the commands below.

## 2. Required Parameter Settings

- `src/experiment.ini`: dataset filenames, GA search settings, and output directories.
- `src/dp_perturb.py`: privacy budgets, clipping thresholds, candidate-pool settings, utility degradation bounds, and utility-experiment run counts.
- `src/Inference_Attack_Compare_Original.py` and `src/Inference_Attack_Compare_Differ.py`: fixed comparison settings for the two inference attacks. Each uses 1,000 tasks and 20 runs, with sequence lengths of 20 for QWS and 40 for Normal. The hypothesis count is 50.

## 3. Running the Code

### 3.1 Generate Reference Solutions

Before running the main experiment, generate the reference solutions for the selected dataset:

```bash
python src/Benchmark.py <dataset>
```

Reference solutions are saved to `src/exact_reference/`. They are required for the main utility experiment and its candidate-pool ablation.

### 3.2 Run the Main PriQoSC Experiment

```bash
python src/OnDemand_GA.py <dataset>
```

Results are saved to `src/results/<dataset>_dp_qosc_results.txt`.

### 3.3 Run Candidate-Pool Parameter Ablation Experiments

```bash
python src/OnDemand_GA.py <dataset> ablation
```

Results are saved to `src/results/<dataset>_delta_optcount_ablation_results.txt`.

### 3.4 Run the Preference Inference Attacks

White-box Inference Attack: compare the released sequence with GA-generated composition templates.

```bash
python src/Inference_Attack_Compare_Original.py <dataset>
```

Black-box Inference Attack: compare released component QoS values directly with candidate constraint upper bounds.

```bash
python src/Inference_Attack_Compare_Differ.py <dataset>
```

Both attacks use Euclidean distance and report PRA and constraint NMAE. Results are saved after each completed run to:

```text
src/results/<dataset>_compare_original_results.txt
src/results/<dataset>_compare_differ_results.txt
```

Completed runs resume automatically. Use `--restart-runs` to restart run-level statistics. For the original attack, use `--length-mode native` to retain the original task lengths; these results are saved separately. QWS and Normal can run in separate terminals and write separate result files.

### 3.5 Run the ClipBound/OptCount Inference Attack Ablation Experiment

```bash
python src/Inference_Attack_ClipBound_OptCount_Ablation.py <dataset>
```

Results are saved under `src/results/`.
