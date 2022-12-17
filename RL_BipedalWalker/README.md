##  RL BipedalWalker

#### Instructions on launching *MDPFuzz* for   Reinforcement Learning models for BipedalWalker


----

#### Notes:

The core component of *MDPFuzz* is in the folder `./rl-baselines3-zoo/fuzz/`.

The RL algorithm is in the folder `./rl-baselines3-zoo`.

The RL model we evaluate is borrowed from these awesome repositories: https://github.com/DLR-RM/rl-baselines3-zoo, https://github.com/DLR-RM/rl-trained-agents, which are under MIT license.

----

#### Setting up environment:

Run the following:
```bash
# Setup environment
conda create -n RLWalk python=3.6.3
conda env update --name RLWalk --file environment_RLWalk.yml
conda activate RLWalk
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .

# Download trained models
cd ./rl-baselines3-zoo
git clone https://github.com/DLR-RM/rl-trained-agents
```

----

#### Fuzz testing:

Check the default path of the model is correct in `./enjoy.py`. 

Run `python enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --em --no-render` to start fuzz testing.

----

#### Root cause:

Run `python Tsne.py` to see the visualization results of projecting states to 2-dimentional spaces using TSNE.

We provide our crash-triggering and normal data in folder `./results/`, and you can also use the states selected by your own and use `Tsne.py` to plot the figures.
