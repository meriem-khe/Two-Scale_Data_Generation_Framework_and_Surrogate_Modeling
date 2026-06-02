# Two-Scale Data Generation Framework and Surrogate Modeling 

This repository contains the complete codebase for a novel framework designed to address data scarcity in multiscale modeling. The core objective of this work is to overcome the high computational costs of traditional two-scale dynamical system simulations by enabling the training of highly efficient data-driven surrogate models.

Because data for these systems is inherently scarce, our approach introduces a systematic pipeline divided into two principal phases: **Data Generation** via physics-based simulation, and **Surrogate Training** for rapid state forecasting.

`![Full Pipeline](figures/general_pipeline.jpg)`

## Methodology Overview

The workflow implemented in this repository is divided into two main parts:

### Part 1: Physics-Based Data Generation

1. **Simulation:** To bridge the data scarcity gap, we propose a structured data generation framework using a sequential pipeline based on the **Finite Element Method (FEM)**. This step generates two high-fidelity datasets capturing behaviors at both the **macroscopic** and **microscopic** scales.

2. **Noise Injection:** To introduce more realism, a post-processing step is performed, which consists of injecting correlated noise into the data.

### Part 2: Surrogate Model Training

Once the datasets are successfully generated, the pipeline moves to the machine learning phase:
1. **Preprocessing:** Data is structured and split for training.
2. **Surrogate Training & Evaluation:** Four distinct machine learning architectures: ARIMA, PROPHET, LSTM, and RANDOM FOREST, are trained on the processed macroscopic dataset as a proof of concept to forecast future system states rapidly, significantly reducing computational cost during inference.

## Repository Structure

The project is organized into dedicated directories within the main folder to ensure a clean workflow:

```text
├── data_generation/          # Scripts and FEM pipelines for macro/micro data generation
├── preprocessing_noise/      # Scripts for correlated noise injection and data normalization
├── surrogate_models/         # Training scripts for the 4 machine learning architectures
├── figures/                  # Generated plots, performance evaluations, and loss curves
├── results/                  # Saved model weights, metrics, and evaluation summaries
└── README.md                 # Project overview and documentation

```

