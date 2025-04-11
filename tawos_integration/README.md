# TAWOS Dataset Integration for SAGE

This directory contains code and resources for integrating the TAWOS (Tabular And Web Object Segmentation) dataset with the SAGE Enterprise Graph RAG system.

## Purpose

The TAWOS dataset will be used for:
1. Training and evaluating NLP components in SAGE
2. Benchmarking performance against state-of-the-art systems
3. Data augmentation to improve model performance on enterprise documents

## Directory Structure

- `preprocessing/`: Scripts for preprocessing TAWOS data
- `evaluation/`: Scripts for evaluating model performance using TAWOS
- `models/`: Fine-tuned models trained on TAWOS data
- `data/`: Processed TAWOS data (not raw data, which should be downloaded separately)

## Getting Started

1. Download the TAWOS dataset from GitHub
2. Run the preprocessing scripts to format the data for SAGE
3. Use the processed data to train or evaluate SAGE components

## Integration Plan

1. **Data Preprocessing**: Convert TAWOS data to match enterprise document formats
2. **Feature Engineering**: Extract relevant features for NLP models
3. **Model Training**: Fine-tune DeepSeek R1 and other models on TAWOS data
4. **Evaluation**: Measure performance metrics on test portions of TAWOS
