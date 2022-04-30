##  RL CARLA

#### Instructions on launching *MDPFuzz* for Reinforcement Learning models for CARLA

----

#### Notes:
The core code of *MDPFuzz* is in `./fuzz/fuzz.py`. 

The RL model we evaluate is borrowed from this awesome repository: https://github.com/valeoai/LearningByCheating.

Part of the `PythonAPI` and the map rendering code is borrowed from the official [CARLA](https://github.com/carla-simulator/carla) repo, which is under MIT license.

----

#### Setting up environment:

Run the following:
```bash
# Setup CARLA
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
mkdir carla_RL_IAs
tar -xvzf CARLA_0.9.6.tar.gz -C carla_RL_IAs
cd carla_RL_IAs
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin
mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg

# Setup environment
conda create -n carla_RL_IAs python=3.5.6

easy_install carla-0.9.6-py3.5-linux-x86_64.egg

cd ../../..

conda env update --name carla_RL_IAs --file environment_carlarl.yml
conda activate carla_RL_IAs

# Download models
wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_only_town01_train_weather.zip
unzip model_RL_IAs_only_town01_train_weather.zip

wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_CARLA_Challenge.zip
unzip model_RL_IAs_CARLA_Challenge.zip
```

----

#### Fuzz testing:
First run `./carla_RL_IAs/CarlaUE4.sh -fps=10 -benchmark -carla-port=3000` to start the CARLA environment.

Run `python benchmark_agent.py --suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --emguide --port=3000` to start fuzzing.

We set a default terminate running time for 1 hour with 50 initial seeds. You may adjust the setting accordingly in `./benchmark/run_benchmark.py`.

----

#### Root cause:
Run `python Tsne.py` to see the visualization results of projecting states to 2-dimentional spaces using TSNE.

We provide our crash-triggering and normal data in folder `./results/`, and you can also use the states selected by your own and then use `Tsne.py` to plot the figures.