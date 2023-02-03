# Running Experiments on GCP

Most experiments were run on a Google Cloud Compute Engine VM with the following specs:

- NVIDIA T4 GPU
- 4 vCPU, 15 GB RAM (n1-standard-4)
- 200 GB disk
- Debian 10 based Deep Learning VM with M102 image

Once you've created the machine and installed the NVIDIA drivers (you will be prompted to do so on your first login) follow the steps below to configure the environment:

1. Download and install anaconda by running the following command:

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
sh Anaconda3-2022.10-Linux-x86_64.sh
```

1. Install the `wilds` package and remove `torch` and `torchvision` (we need to install an older version of these packages because of issues with CUDA dynamic libraries). In order to run all the CGD experiments you will need to use our custom version of wilds, available in the `wilds` folder.

```sh
# Create a conda environment if you haven't done so already. Wilds suggests Python 3.8.5
conda create -n wilds python=3.8.5
conda activate wilds

# Install dependencies
# Install the modified version of wilds
pip install -e ./wilds
# Install additional libraries required by wilds
pip install transformers
pip install wandb # only if you're using WandB
# Remove torch and torchvision
pip uninstall torch
pip uninstall torchvision
```

3. Install `torch==1.12.1` and the related dependencies

```sh
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

4. Install `torch-geometric`

```sh
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric
```

Now you should be ready to run experiments!

### Downloading datasets

The WILDS dataset are quite large (up to 50GB!) and downloading them directly from the source may be slow (and cost $$$). You can pre-download them on your machine (or on a CPU only machine) and download them from Google Drive with `gdown`

```sh
pip install gdown
# Download the file from Drive (link sharing must be on). You can find the file id in the URL
gdown "https://drive.google.com/uc?id=<file-id>"
```

### Running Processes

In order to run processes in the background (and avoid them crashing when the ssh session is terminated) run them as

```sh
nohup <command> &
```

And then monitor their stdout with the following command (the PID will be printed out when the process starts)

```sh
tail -f /proc/<pid>/fd/1
```
