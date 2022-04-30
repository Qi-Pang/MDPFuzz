##  ACAS Xu (DNN)

#### Instructions on launching *MDPFuzz*
----

#### Setting up environment:

Run the following:
```bash
conda create -n acas python=3.7.9
conda env update --name acas --file environment_ACAS.yml
conda activate acas
```

----

#### Notes:
The core code of *MDPFuzz* is in `./fuzz/fuzz.py`. 

`./simulate.py` is the main simulation code for ACAS Xu.

All models including ACAS Xu models and models after repair are inside `./models`.

----

#### Fuzz testing:
Run `python simulate.py` to start fuzz testing.

We set a default terminate running time for 1 hour with 50 initial seeds. You may adjust the time to 6 hours with 500 initial seeds by `python simulate.py --terminate 6 --seed_size 500`.

----

#### Repair models:
After fuzz testing, we can repair the models using the crash-triggering inputs found by *MDPFuzz*.

The corresponding code is `./repair.py`.

Run `mkdir checkpoints` and `python repair.py` to repair the models using the crash-triggering state sequences found by *MDPFuzz*, the repaired model will be stored in the folder `checkpoints`.

To evaluate the performance of the repaired models, just replace the corresponding original models with the repaired models, and then run fuzz testing again.

----

#### Root cause:
Run `python Tsne.py` to see the visualization results of projecting states to 2-dimentional spaces using TSNE.

We provide our crash-triggering and normal data in folder `./results/`, and you can also use the states selected by your own and use `Tsne.py` to plot the figures.