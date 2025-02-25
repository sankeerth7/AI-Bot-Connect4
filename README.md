# AI-Bot-Connect4

This repository contains the source code and documentation for an AI bot designed to play Connect Four. The project explores and compares multiple deep learning architectures—including Convolutional Neural Networks (CNNs), Transformer models, and a hybrid CNN-Transformer approach—to predict the optimal move from any given board state. The models are trained on high-quality, MCTS-driven data and deployed as an interactive web application using Anvil, Docker, and AWS.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Generation & Preparation](#data-generation--preparation)
- [Model Architectures](#model-architectures)
- [Training & Optimization](#training--optimization)
- [Deployment](#deployment)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Setup & Installation](#setup--installation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

The goal of this project was to develop a high-performing AI capable of predicting the best move in Connect Four from any board configuration. By experimenting with various deep learning models and leveraging advanced techniques like Monte Carlo Tree Search (MCTS) for data generation, we gained valuable insights into model design, hyperparameter tuning, and deployment strategies.

## Features

- **Multiple Architectures:** Implemented and compared CNN, Transformer, and hybrid models.
- **Data Generation:** Utilized MCTS and random move generation with board mirroring to create a diverse dataset (~2 million examples).
- **Optimized Training:** Employed adaptive learning rate schedules, early stopping, and hyperparameter tuning using Keras Tuner.
- **Scalable Deployment:** Containerized the application with Docker and hosted it on AWS via an Anvil-powered web interface.

## Data Generation & Preparation

- **MCTS-Driven Data:** Generated training examples using Monte Carlo Tree Search to simulate future moves and evaluate optimal plays.
- **Data Augmentation:** Increased dataset size through board mirroring and random move initialization.
- **Preprocessing:** Standardized board states, applied one-hot encoding for target labels, and split data into training and validation sets.

## Model Architectures

1. **CNN Model:** 
   - Custom convolutional kernels to detect horizontal, vertical, and diagonal patterns.
   - Layers include convolution, batch normalization, max pooling, and fully connected layers.
   - Achieved up to 77% training accuracy.

2. **Transformer Model:** 
   - Utilized overlapping patch extraction, positional embeddings, and multi-head self-attention.
   - Captured global board relationships, though it was outperformed by the CNN in this specific task.

3. **Hybrid Model:** 
   - Combined the spatial pattern recognition of CNNs with the long-range dependency handling of Transformers.
   - Integrated custom layers for tokenization and employed a warmup plus cosine decay learning rate schedule.

## Training & Optimization

- **Hyperparameter Tuning:** Used Keras Tuner with Hyperband strategy to optimize architecture and training parameters.
- **Training Strategy:** Leveraged early stopping and adaptive learning rate schedules (warmup and cosine decay) to improve convergence and prevent overfitting.
- **Performance:** CNN model achieved the best performance, while the Transformer and hybrid models provided additional insights into model design trade-offs.

## Deployment

- **Web Application:** Built using [Anvil](https://anvil.works/) for an interactive front-end.
- **Containerization:** Deployed using Docker to ensure consistency across environments.
- **Cloud Hosting:** Hosted on AWS, enabling scalable and robust performance for real-time gameplay.

