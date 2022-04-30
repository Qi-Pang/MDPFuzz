# End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances

This repo contains the inference code and the weights of our [**paper**](https://arxiv.org/abs/1911.10868)
accepted at CVPR 2020. 
It's a fork of the repository [**Learning by Cheating**](https://github.com/dianchen96/LearningByCheating) 
from which we just kept all the code related to the evaluation on the standard CARLA
benchmark and on the new released No-Crash benchmark.

### Installation
We provide a script to install every dependencies needed and download our weights.

```bash
# Download CARLA 0.9.6
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
mkdir carla_RL_IAs
tar -xvzf CARLA_0.9.6.tar.gz -C carla_RL_IAs
cd carla_RL_IAs

# Download maps
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin
mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/

# Install carla client
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg
easy_install carla-0.9.6-py3.5-linux-x86_64.egg
cd ../../..

# Create conda environment
conda env create -f environment.yml
conda activate carlaRL

# Download model checkpoints trained only on Town01/training weathers
wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_only_town01_train_weather.zip
unzip model_RL_IAs_only_town01_train_weather.zip

# Download model checkpoints used for CARLA challenge
wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_CARLA_Challenge.zip
unzip model_RL_IAs_CARLA_Challenge.zip
