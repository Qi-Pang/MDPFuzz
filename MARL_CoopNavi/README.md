##  MARL Coop Navi

#### Instructions on launching *MDPFuzz* for   Multi-Agent Reinforcement Learning models for Coop-Navi

----

#### Notes:

The core component of *MDPFuzz* is in the folder `./maddpg/experiments/fuzz/`.

The MARL algorithm is in the folder `./maddpg`.

The Coop Navi environment is in the folder `./multiagent-prticle-envs`.

The MARL model we evaluate is borrowed from this awesome repository: https://github.com/openai/maddpg, which is under MIT license.

The Coop Navi environment is installed according to this repository: https://github.com/openai/multiagent-particle-envs, which is under MIT license.

----

#### Setting up environment:

```bash
conda create -n MARL python=3.5.4
conda env update --name MARL --file environment_MARL.yml
conda activate MARL

cd ./maddpg
pip install -e .
cd ../multiagent-particle-envs
pip install -e .
cd ../maddpg/experiments/
```
----

#### Train the model:

We first train the MARL model according to the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf) and the repository https://github.com/openai/maddpg.

We provide our trained model in `./maddpg/checkpoints`. Users can also follow the [instructions](./README-maddpg.md) to train their own models.


----

#### Fuzz testing:

Check the default path of the model is correct in `./testing.py`. 

Run `python testing.py` to start fuzz testing.

----

#### Root cause:

Run `python Tsne.py` to see the visualization results of projecting states to 2-dimentional spaces using TSNE.

We provide our crash-triggering and normal data in folder `./results/`, and you can also use the states selected by your own and use `Tsne.py` to plot the figures.
