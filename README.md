# Evolving Modular Soft Robots without Explicit Inter-Module Communication using Local Self-Attention
This is the official repository for the GECCO (Genetic and Evolutionary Computation Conference, 2022) paper

**<a href="">Evolving Modular Soft Robots without Explicit Inter-Module Communication using Local Self-Attention</a>**
<br>
<a>Federico Pigozzi</a>,
<a>Yujin Tang</a>,
<a>Eric Medvet</a>,
<a>David Ha</a>
<br>

hosting all the code necessary to replicate the experiments. This work was carried out at the Evolutionary Robotics and Artificial Life Laboratory (ERALLab) at the University of Trieste (Italy). More videos available at this [link](https://softrobots.github.io).

<div align="center">
<img src="teaser.gif"></img>
</div>

## Scope
By running
```
java -cp libs:JGEA.jar:libs/TwoDimHighlyModularSoftRobots.jar:target/AttentionVSRs.jar world.units.erallab.Main {args}
```
where `{args}` is a placeholder for the arguments you provide (see below), you will launch an evolutionary optimization for evolving jointly the controller (a self attention-based artificial neural network) of Voxel-based Soft Robots (VSRs). At the same time, evolution metadata will be saved inside the `output` folder. The project has been tested with Java `14.0.2`.

## Structure
* `src` contains all the source code for the project;
* `libs` contains the .jar files for the dependencies (see below);
* `Data_Analysis_Notebook.ipynb` is a Jupyter Notebook (Python) with some routines and starter code to perform analysis on the evolution output files.

## Dependencies
It relies on:
* [JGEA](https://github.com/ericmedvet/jgea), for the evolutionary optimization;
* [2D-VSR-Sim](https://github.com/ericmedvet/2dhmsr), for the simulation of VSRs.

The relative jars have already been included in the directory `libs`. See `pom.xml` for more details on dependencies.

## Usage
This is a table of possible command-line arguments:
Argument       | Type                                               | Optional (yes/no) | Default
---------------|----------------------------------------------------|-------------------|-------------------------
exp            | {baseline,attention}                               | no                | -
config         | {none-homo,neumann-homo,none-4-8-8-homo|homo-tanh} | no                | -
shape          | {biped-4x3,biped-6x4,comb-7x2,comb-14x2}           | no                | -
seed           | integer                                            | no                | -
threads        | integer                                            | yes               | # available cores on CPU

where {...} denotes a finite and discrete set of possible choices for the corresponding argument. The description for each argument is as follows:
* exp: the controller architecture to experiment with, either an MLP baseline or self attention-based.
* config: `none-homo` for MLP baseline without communication, `none-neumann` for fixed-communication MLP baseline, and `none-4-8-8-homo|homo-tanh` for self-attention without communication.
* shape: the VSR shape to experiment with.
* seed: the random seed for the experiment.
* threads: the number of threads to perform evolution with. Defaults to the number of available cores on the current CPU. Parallelization is taken care by JGEA and implements a distributed fitness assessment.

## Bibliography
Please cite as:
```
@article{pigozzi2022evolving,
  title={Evolving Modular Soft Robots without Explicit Inter-Module Communication using Local Self-Attention},
  author={Pigozzi, Federico and Tang, Yujin and Medvet, Eric and Ha, David},
  journal={arXiv preprint arXiv:2204.06481},
  year={2022}
}
```
