# Generative Adversarial Network (GAN) Implementation

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN). The model is designed to generate realistic images by training a generator and discriminator in an adversarial manner.

## Architecture Overview

### Generator
The generator transforms random noise vectors into synthetic images through a series of transposed convolutional layers:

### Discriminator
The discriminator evaluates whether images are real or generated through convolutional downsampling:

## Requirements

- Python 
- PyTorch 
- torchvision
- numpy
- matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/viswa0028/anime-image-generation.git
cd anime-image-generation
pip install -r requirements.txt
```

## Dataset
The dataset is an official Kaggle directory of Anime Images - (Kaggle Dataset)[https://www.kaggle.com/code/sachinrajput17/gans-for-anime-face-dataset/input]

## Results

Generated samples will be saved in the `samples/` directory during training. The final model weights are stored in the `checkpoints/` directory.

### Sample Generation Progress

![Training Progress](assets/training_progress.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The architecture is inspired by [DCGAN paper](https://arxiv.org/abs/1511.06434)
