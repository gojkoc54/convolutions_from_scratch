# Implementation of the paper "Towards Learning Convolutions From Scratch"
**(Reproducibility Challenge, Fundamentals of Inference and Learning EE-411, EPFL)**


**DISCLAIMER** - There is no public implementation of this paper, 
so the whole code base in this repository was designed from scratch 
(as the convolutions in the paper... dad joke, I know)


## Structure of the project

```
convolutions_from_scratch/
    |
    |-- src/
    |   |
    |   |-- scripts/
    |   |   |-- run_train.sh
    |   |   |-- run_figure_3.sh
    |   |
    |   |-- models.py
    |   |-- optim.py
    |   |-- train.py
    |   |-- utils.py
    |   |-- results_visualization.ipynb
    |   
    |-- checkpoints/
    |   |-- cifar10/
    |   |-- svhn/
    |   |-- figure_3/
    |
    |-- plots/
    |
    |-- environment.yml
    |
    |-- report.pdf
    |
    |-- README.md
```

* ```results_visualization.ipynb```
    * pre-compiled jupyter notebook containing the visualization process of our results
    * used to generate all the figures in our report
* ```optim.py```
    * python file containing the implementation of the $\beta$-LASSO optimizer
* ```scripts/run_train.sh```, ```scripts/run_figure_3.sh```
    * bash scripts used for running all the experiments in our project
* ```models.py```
    * python file containing implementations of the architectures 
    used in this project (S-FC, S-LOCAL, ...)
* ```utils.py```    
    * python file containing functions that are used in the training 
    pipeline of the project - training, testing, checkpointing, ...
* ```train.py```
    * main script for training; it combines all the necessary functions and 
    performs the whole training pipeline
* ```checkpoints```
    * directory containing the pre-trained models (checkpoints) and 
    metric trackers for all experiments
    * it's generated during training and later used for 
    the visualization process


## Running the code

* Ready-to-use notebook for visualizing our results 
```src/results_visualizations.ipynb```

* Set up the conda environment (GPU required!)

*DISCLAIMER* - the good old 'it worked on my machine' ... 
We hope it works on your as well, but can't guarantee :) 

```
conda env create -f environment.yml
conda activate convs_from_scratch
```

* Run the training scripts

```
cd src/scripts
chmod 777 run_train.sh run_figure_3.sh
./run_train.sh
./run_figure_3.sh
```



## Team info

* Team members:
    * Gojko Cutura - gojko.cutura@epfl.ch
    * Soroush Mehdi - soroush.mehdi@epfl.ch
    * Khashayar Najafi - khashayar.najafi@epfl.ch
    

