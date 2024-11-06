# Installation instructions 

```bash
git clone git@github.com:marmotlab/contractual_gifting.git
cd contractual_gifting
git checkout April_Presenting 
cd Cooperation\ Metric\ Checks/marl-benchmark/
```


```bash
conda config --append channels conda-forge
conda create -n marl_benchmark python==3.6.1
conda activate marl_benchmark
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install on-policy package
cd on-policy
pip install -e .


pip install wandb

pip install -r requirements.txt

pip uninstall grpcio
pip install grpcio

pip uninstall -y enum34

pip uninstall setuptools # run this a few times 
pip install -U pip setuptools


pip uninstall -y enum34 # Do this again if cannot install smac
pip install -U protobuf

pip install enum34
pip install -U absl-py

pip install git+https://github.com/oxwhirl/smacv2.git

pip install networkx


pip uninstall matplotlib
pip install matplotlib==2.2.5
pip install -U filelock
pip install -U importlib-metadata

pip uninstall -y wandb # Do this again if cannot install smac
pip install wandb==0.10.5

```

Make sure to install maps from here [Release SMACv2 Maps · oxwhirl/smacv2 · GitHub](https://github.com/oxwhirl/smacv2/releases/tag/maps)


Run Commands

```bash

conda activate marl_benchmark 
cd marl-benchmark/onpolicy/scripts/train_smacv2_scripts
# cd onpolicy/scripts/train_smacv2_scripts/
bash train_terran_5v5.sh


```


# Run instructions

```bash
pkill -f SC2
cd Cooperation\ Metric\ Checks/


cd onpolicy/scripts/train_smacv2_scripts/

echo $SC2PATH

```

# Test Instructions

## Running
### Test file 

```bash
screen -R PamTesting
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark
bash pam_train_smac_baneling_Positioning_quicktest.sh

```

### 5m_vs_6m	
https://github.com/oxwhirl/smac/blob/d6aab33f76abc3849c50463a8592a84f59a5ef84/docs/smac.md

```bash

screen -R PamMarines
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

bash pam_train_smac_5m_vs_6m_Marines.sh
```

### Kiting map - 3s_vs_3z -- mappo algo

```bash

screen -R PamKiting
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

bash pam_train_smac_3s_vs_3z_Kiting.sh
```
### MicroFire map 
```bash

screen -R PamAlternateFire
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark
bash pam_train_smac_2m_vs_1z_MF.sh
```
### baneling positioning 

```bash

screen -R PamPosition     
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark
bash pam_train_smac_baneling_Positioning.sh
```
### Good old 3m with mappo ---- 

```bash
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

bash pam_train_smac_3m_mappo.sh  # This one has mappo

# bash pam_train_smac_3m.sh # this is rmappo

```

### Wall off 
```bash

screen -R PamCorridor
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

bash pam_train_smac_corridor.sh

```


### SMAC 2V2
cd marl-benchmark/onpolicy/scripts/train_smacv2_scripts/
conda activate marl_benchmark
bash pam_wandb_terran5v5noTrainSI.sh


### Env tests

conda activate marl_benchmark
bash pam_testing_terran5v5grpSI.sh

conda activate marl_benchmark
bash pam_wandb_terran5v5grpSI.sh



https://stackoverflow.com/questions/8987037/how-to-kill-all-processes-with-a-given-partial-name

## Loading the model and running 

### 5m_vs_6m	
https://github.com/oxwhirl/smac/blob/d6aab33f76abc3849c50463a8592a84f59a5ef84/docs/smac.md

```bash

screen -R PamMarines

cd marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark
bash pam_load_smac_5m_vs_6m_Marines.sh
```

### Kiting map - 3s_vs_3z -- mappo algo

```bash

screen -R PamKiting
cd marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark

bash pam_load_smac_3s_vs_3z_Kiting.sh
```
### MicroFire map 
```bash

screen -R PamAlternateFire
cd marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark
bash pam_load_smac_2m_vs_1z_MF.sh
```

### Good old 3m 

```bash
screen -R Pam3m

cd marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark

bash pam_load_smac_3m_mappo.sh  # This one has mappo
```

### baneling positioning ---- this is not prepped yet

```bash

screen -R PamPosition     
cd marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark
bash pam_load_smac_baneling_Positioning.sh
```

# Installation Instructions for MarlBenchmark

[GitHub - marlbenchmark/on-policy: This is the official implementation of Multi-Agent PPO (MAPPO).](https://github.com/marlbenchmark/on-policy)
I have cuda 11.5 so decided to do pytorch 1.12.0 which has CUDA 11.3 stuff but that didn't work.

These installation instructions

``` bash
conda config --append channels conda-forge
conda create -n marl_benchmark python==3.6.1
conda activate marl_benchmark
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install on-policy package
cd on-policy
pip install -e .


pip install wandb

pip install -r requirements.txt


pip uninstall -y enum34

pip uninstall setuptools # run this a few times 
pip install -U pip setuptools

pip uninstall -y enum34 # Do this again if cannot install smac
pip install git+https://github.com/oxwhirl/smacv2.git

pip install networkx
```

Make sure to install maps from here [Release SMACv2 Maps · oxwhirl/smacv2 · GitHub](https://github.com/oxwhirl/smacv2/releases/tag/maps)


Run Commands

```bash

conda activate marl_benchmark
cd marl-benchmark/onpolicy/scripts/train_smacv2_scripts
# cd onpolicy/scripts/train_smacv2_scripts/
bash train_terran_5v5.sh


```

# Example of how to do this

### 3M Mappo

Start with training
```bash
pkill -f SC2
screen -R Pam3m

cd Cooperation\ Metric\ Checks/
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark
export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"
bash pam_train_smac_3m_mappo.sh  # This one has mappo

# bash pam_train_smac_3m.sh # this is rmappo
bash pam_train_smac_3m_mappo_SuperLong.sh
```

put the trained policy model path into the pam_load_smac_3m_mappo.sh file

```bash
screen -R Pam3m

cd Cooperation\ Metric\ Checks/
cd marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark
export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"

bash pam_load_smac_3m_mappo.sh  # This one has mappo

bash pam_load_smac_3m_mappo_10M.sh

```

Take the model paths of both SI model (the folders should be seperate, and you put path of folder in and it will look for the correct network 1 2 3 etc) and trained policy model path into Cooperation Metric Checks/marl-benchmark/onpolicy/scripts/comparison_scripts/pam_compare_smac_3m_mappo.sh

```bash
screen -R Pam3m

export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"
cd marl-benchmark/onpolicy/scripts/comparison_scripts/
conda activate marl_benchmark

bash pam_compare_smac_3m_mappo\(Load_Policy\).sh 

# bash pam_compare_smac_3m_mappo\(NewPolicy\).sh  

```

rm -rf run1
rm -rf run2
rm -rf run3
rm -rf run4
rm -rf run5
rm -rf run6
rm -rf run7
rm -rf run8

## 6h_vs_8z

Start with training

```bash
screen -R Pam6h
export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"
cd Cooperation\ Metric\ Checks/marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

pkill -f SC2
bash pam_train_smac_6h_vs_8z.sh  # This one has mappo

```


put the trained policy model path into the pam_load_smac_3m_mappo.sh file

```bash
screen -R Pam6h

cd Cooperation\ Metric\ Checks/marl-benchmark/onpolicy/scripts/load_smac_scripts/
conda activate marl_benchmark

bash pam_load_smac_6h_vs_8z.sh  # This one has mappo

```

Take the model paths of both SI model (the folders should be seperate, and you put path of folder in and it will look for the correct network 1 2 3 etc) and trained policy model path into Cooperation Metric Checks/marl-benchmark/onpolicy/scripts/comparison_scripts/pam_compare_smac_3m_mappo.sh

```bash
screen -R Pam6h

cd Cooperation\ Metric\ Checks/marl-benchmark/onpolicy/scripts/comparison_scripts/
conda activate marl_benchmark

bash pam_compare_smac_6h_vs_8z-Load.sh


```

### 3s5z

```bash
screen -R Pam3s5z
export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"
cd Cooperation\ Metric\ Checks/marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

pkill -f SC2
bash pam_train_smac_3s_vs_5z.sh  # This one has mappo

```

TODO load

```bash
screen -R Pam6h

cd Cooperation\ Metric\ Checks/marl-benchmark/onpolicy/scripts/comparison_scripts/
conda activate marl_benchmark

bash pam_compare_smac_3svs5z_mappo.sh


```

### MMM2

```bash
screen -R PamMMM2
export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"
cd Cooperation\ Metric\ Checks/marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

pkill -f SC2
bash pam_train_smac_MMM2.sh  # This one has mappo

```

### QuickTester

```bash
screen -R Pam3m

export PYTHONPATH="${PYTHONPATH}:/home/marmot/Pamela/contractual_gifting"
cd marl-benchmark/onpolicy/scripts/train_smac_scripts/
conda activate marl_benchmark

bash pam_train_smac_3m_mappo_quicktest.sh  # This one has mappo

# bash pam_train_smac_3m.sh # this is rmappo

```
