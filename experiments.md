# Running experiments

This document shows how each experiment in the reproducibility report can be replicated. Before running experiments set up an appropriate environment as shown in `env_setup.md`.

Only the variable parameters are specified as command line arguments, for the full configuration for each dataset refer to `wilds/examples/configs/datasets.py`.

## Qualitative experiments

### 4.1 - Label Noise

1. Create a data folder for the dataset

```sh
mkdir -p data/noisy_2feature
```

2. Run `Group-DRO` and `CGD` **Simple** experiments. Each experiment should be run for 6 seeds. We used seeds `[0, 5, 8, 13, 42, 3]`

```sh
# Group-DRO
python wilds/examples/run_expt.py \
  --dataset noisy_2feature \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/noisy_2feature/groupDRO/run:2:seed:0 \
  --seed 0
```

```sh
# CGD
python wilds/examples/run_expt.py \
  --dataset noisy_2feature \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/noisy_2feature/CG/run:2:seed:0 \
  --seed 0 \
  --cg_C 0 \
  --cg_step_size 0.05
```

3. Create a data folder for the **Simple-MNIST** dataset

```sh
mkdir -p data/noisy_mnist
```

4. Run `Group-DRO` and `CGD` **Simple-MNIST** experiments. Each experiment should be run for 3 seeds. We used seeds `[0, 13, 42]`

```sh
# Group-DRO
python wilds/examples/run_expt.py \
  --dataset noisy_mnist \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/noisy_mnist/groupDRO/run:2:seed:0 \
  --seed 0
```

```sh
# CGD
python wilds/examples/run_expt.py \
  --dataset noisy_mnist \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/noisy_mnist/CG/run:2:seed:0 \
  --seed 0 \
  --cg_C 0 \
  --cg_step_size 0.05
```

### 4.2 - Uneven Inter-group Similarity - Rotation

1. Create a data folder for the dataset

```sh
mkdir -p data/rot_simple
```

2. Run `Group-DRO` and `CGD` experiments. Each experiment should be run for 6 seeds. We used seeds `[0, 5, 8, 13, 42, 3]`

```sh
# Group-DRO
python wilds/examples/run_expt.py \
  --dataset rot_simple \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/rot_simple/groupDRO/run:2:seed:0 \
  --seed 0
```

```sh
# CGD
python wilds/examples/run_expt.py \
  --dataset rot_simple \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/rot_simple/CG/run:2:seed:0 \
  --seed 0 \
  --cg_C 0 \
  --cg_step_size 0.05
```

3. Create a data folder for the **Simple-MNIST** dataset

```sh
mkdir -p data/rot_mnist
```

4. Run `Group-DRO` and `CGD` **Simple-MNIST** experiments. Each experiment should be run for 3 seeds. We used seeds `[0, 13, 42]`

```sh
# Group-DRO
python wilds/examples/run_expt.py \
  --dataset rot_mnist \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/rot_mnist/groupDRO/run:2:seed:0 \
  --seed 0
```

```sh
# CGD
python wilds/examples/run_expt.py \
  --dataset rot_mnist \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/rot_mnist/CG/run:2:seed:0 \
  --seed 0 \
  --cg_C 0 \
  --cg_step_size 0.05
```

### 4.3 - Spurious Correlations

1. Create a data folder for the dataset

```sh
mkdir -p data/spu_2feature
```

2. Run `Group-DRO` and `CGD` experiments. Each experiment should be run for 6 seeds. We used seeds `[0, 5, 8, 13, 42, 3]`

```sh
# Group-DRO
python wilds/examples/run_expt.py \
  --dataset spu_2feature \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/spu_2feature/groupDRO/run:2:seed:0 \
  --seed 0
```

```sh
# CGD
python wilds/examples/run_expt.py \
  --dataset spu_2feature \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/spu_2feature/CG/run:2:seed:0 \
  --seed 0 \
  --cg_C 0 \
  --cg_step_size 0.05
```

3. Create a data folder for the **Simple-MNIST** dataset

```sh
mkdir -p data/spu_mnist
```

4. Run `Group-DRO` and `CGD` **Simple-MNIST** experiments. Each experiment should be run for 3 seeds. We used seeds `[0, 13, 42]`

```sh
# Group-DRO
python wilds/examples/run_expt.py \
  --dataset spu_mnist \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/spu_mnist/groupDRO/run:2:seed:0 \
  --seed 0
```

```sh
# CGD
python wilds/examples/run_expt.py \
  --dataset spu_mnist \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/spu_mnist/CG/run:2:seed:0 \
  --seed 0 \
  --cg_C 0 \
  --cg_step_size 0.05
```

## Quantitative experiments

### non-WILDS datasets

Each non-WILDS dataset was run for three seeds `[0, 13, 42]` and for three algorithms: `ERM`, `CGD` and `Group-DRO`.

#### CMNIST

```sh
# ERM
python wilds/examples/run_expt.py \
  --dataset cmnist \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/cmnist/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset cmnist \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/cmnist/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset cmnist \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/cmnist/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 0 \
  --cg_step_size 0.05
  --download
```

#### WaterBirds

```sh
# ERM
python wilds/examples/run_expt.py \
  --dataset waterbirds \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/waterbirds/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset waterbirds \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/waterbirds/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset waterbirds \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/waterbirds/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 0 \
  --cg_step_size 0.05
  --download
```

#### CelebA

```sh
# ERM
python wilds/examples/run_expt.py \
  --dataset celebA \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/celebA/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset celebA \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/celebA/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset celebA \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/celebA/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 2 \
  --cg_step_size 0.05
  --download
```

#### MultiNLI

```sh
# ERM
python wilds/examples/run_expt.py \
  --dataset multiNLI \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/multiNLI/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset multiNLI \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/multiNLI/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset multiNLI \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/multiNLI/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 2 \
  --cg_step_size 0.05 \
  --download
```

### WILDS datasets

#### Camelyon17

```sh
# ERM
python wilds/examples/run_expt.py \
  --dataset camelyon17 \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/camelyon17/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset camelyon17 \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/camelyon17/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset camelyon17 \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/camelyon17/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 0 \
  --cg_step_size 0.05 \
  --download
```

#### PovertyMap

```sh
python wilds/examples/run_expt.py \
  --dataset poverty \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/poverty/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset poverty \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/poverty/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset poverty \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/poverty/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 0 \
  --cg_step_size 0.05 \
  --download
```

#### FMoW

```sh
python wilds/examples/run_expt.py \
  --dataset fmow \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/fmow/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset fmow \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/fmow/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset fmow \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/fmow/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 0 \
  --cg_step_size 0.2
  --download
```

#### CivilComments

```sh
python wilds/examples/run_expt.py \
  --dataset civilcomments \
  --algorithm ERM \
  --root_dir data \
  --progress_bar \
  --log_dir logs/civilcomments/ERM/run:1:seed:42 \
  --seed 42 \
  --download

# Group-DRO
python wilds/examples/run_expt.py \
  --dataset civilcomments \
  --algorithm groupDRO \
  --root_dir data \
  --progress_bar \
  --log_dir logs/civilcomments/groupDRO/run:1:seed:42 \
  --seed 42 \
  --download

# CGD
python wilds/examples/run_expt.py \
  --dataset civilcomments \
  --algorithm CG \
  --root_dir data \
  --progress_bar \
  --log_dir logs/civilcomments/CG/run:1:seed:42 \
  --seed 42 \
  --cg_C 0 \
  --cg_step_size 0.05
  --download
```
