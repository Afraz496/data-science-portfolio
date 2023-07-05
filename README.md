# Data Science Portfolio

This portfolio encompasses 5 types of Data Science projects:

1. Tabular Datasets (Housing Prices, Stock Prices etc)
2. Tabular Datasets with Class Imbalance
3. Computer Vision
4. Natural Language Processing (with a focus on sentiment extraction)
5. Neural Networks and High Performance Computing

This README serves as a guide through the projects, their relevant datasets and some decisions surrounding the Machine Learning exercises in each projects. The README will also cover how to use this repository if you wish to implement these projects on your own but please be sure to cite the creators of the repository: Afraz Arif Khan and Aamir Zargham.

## Setting up environment

The environment includes both a `Dockerfile` in `.devcontainer` and an `environment.yml`. The Docker file can be used to set up the dev environment to ensure reproducibility and streamline moving project into production down the line. In order to use the docker file to generate a container use the VS Code [guide](https://code.visualstudio.com/docs/containers/overview). Note that the file is already set up so after installing VS code and Docker desktop there should be a prompt when you open the project folder to open the folder in the specified container.

In project directory run the following:
```bash
conda env create -f environment.yml
```
This creates an environment named `mlenv` (note you can change the environment name by editing the `environment.yml` file). To activate the virtual environment run:

```bash
conda activate mlenv
```

Note that in osx you may instead need to run:
```bash
source activate mlenv
```

for more information on managing virtual environments see this [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).

**Note**: If you install more dependencies  please update the yml file and run the command before activating the virtual environment:

```bash
conda env update --name mlenv --file environment.yml --prune
```

## Project 1: Tabular Datasets

Tabular Datasets form the backbone of all Machine Learning problems. The terminology for this section 'tabular' is deliberately ambiguous. We will examine Regression and Classification problems but will keep other parameters consistent (like class imblanace) to make less complex machine learning problems. The goal is to examine ideal datasets under relatively ideal conditions and build the backbone of Machine Learning Pipelines.

### [Housing Prices for Kaggle Learners](https://www.kaggle.com/competitions/home-data-for-ml-course/data)

