# Audio Recognition Model

## Overview
This project addresses the Z by HP Unlocked Challenge: to develop a machine learning model and accompanying code capable of counting the number of Capuchinbird calls within a given audio clip.
It was a great project in learning how to interact with audio data and signal processing.

## Methods
- I converted the audio data to a spectogram to get the Fast Fourier Transform of the signals.
- I then passed the data through a Convolutional Deep Neural Network to analyze the given capuchinbird calls (ground truth) and then validate and evaluate the remaining data

## Dependencies
```bash
pip install tensorflowio
```
