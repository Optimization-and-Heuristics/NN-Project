# Neural Network Engine

This project implements a neural network engine from scratch, using only Python and `numpy`, optimized without advanced frameworks like PyTorch or TensorFlow. The architecture is based on modularity principles and heuristic optimization, allowing customization and analysis of key components such as activation functions, optimizers, and loss functions. The attached **Documentation** details the theoretical development, methodology, and experimental results.

## Table of Contents
1. [Description](#description)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Experiments and Results](#experiments-and-results)
6. [Future Plans](#future-plans)
7. [Contributions](#contributions)

## Description

This project focuses on building a neural network engine that applies heuristic optimization algorithms such as **Gradient Descent (GD)** and **Adam**, evaluating their impact on network performance. The model is tested on **MNIST** and **Fashion MNIST** datasets, assessing the generalization capability and the effectiveness of the engine in optimizing parameters without relying on complex external tools.

## Project Structure

The modular structure of the project facilitates the customization and reuse of specific components, allowing adjustments in configuration and detailed analysis of the model's performance. The main modules include:

- **Documentation:** Technical documentation that delves into the architecture and the experiments conducted.
- **Code:** Implementation organized in modules:
  - `NeuronalNetwork.py`: Main class for data flow and learning through backpropagation.
  - `DenseLayer.py`: Implements dense layers with weight initialization, forward pass, and backpropagation.
  - `LossFunction.py`: Loss functions and gradient calculation.
  - `ActivationFunction.py`: Activation functions (ReLU, Softmax) and their gradients.
  - `Optimizer.py`: GD and Adam optimizers.
  - `Trainer.py`: Training and evaluation control.
  - `Test.py`: Evaluation and visualization of results.
  - `NN.ipynb`: Notebook documenting experiments and tests.

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/user/neural-network-engine.git
    ```

2. Install the required dependencies:

    ```bash
    pip install numpy pandas
    ```
    
3. Unzip the data located on datos:

    ```bash
    for file in *.zip; do
    if [ -f "$file" ]; then
        unzip "$file" -d ./
    else
        echo "No zip files found."
    fi
    done
    ```

3. (Optional) Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

## Usage

To run and visualize training and evaluation results, follow these steps:

1. Start **Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

2. Open the `NN.ipynb` notebook, which includes documented tests on MNIST and Fashion MNIST datasets.

3. Adjust parameters (layers, learning rate, etc.) in the notebook to conduct your own experiments.

## Experiments and Results

Experiments were conducted on learning rates and layer configurations to evaluate performance. For the **MNIST** dataset, it was found that a network with one hidden layer optimized with **Adam** at a learning rate of **0.005** achieved a validation accuracy of **97.0%**. For **Fashion MNIST**, the optimal rate was also 0.005, reaching **89.6%** accuracy.

## Future Plans

- **Implementation of CNNs**: To improve image classification performance, it is proposed to include convolutional neural networks (CNNs) that capture spatial patterns.
- **New Optimizers**: Add RMSprop, AdaGrad, and AdaDelta to improve convergence in different datasets.
- **Expansion of Datasets**: Explore more complex datasets to evaluate the generalization capability of the engine.

## Contributions

Contributions are welcome. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-improvement`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-improvement`).
5. Open a **Pull Request**.

## Team Members:
- [Nicole María Ortega Ojeda](https://github.com/nnicoleortegaa)
- [Oscar Rico Rodríguez](https://github.com/orr21)
- [Radosława Żukowska](https://github.com/radoslawazukowska)
