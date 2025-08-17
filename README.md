# Audio Recognition Model

## Overview
This project is a solution to the **Z by HP Unlocked Challenge** to develop a machine learning model and accompanying code capable of counting the number of Capuchinbird calls within a given audio clip.

Project aimed and focused on audio data and signal processing.

## Technical Skills
- TensorFlow                                     
- Convolutional neural networks                 
- Sequential model architecture                  
- Binary classification                           
- Signmoid activation                          
- Short TIme Fourier Transform                   
- Audio signal processing                         
- Model training and hyperparameter tuning        
- Loss functions: (Binary Cross-Entropy) and optimization (Adam) 
- Performance metrics (Precision, Recall)       


## How it works
1. Data Loading & Signal Processing

- Built load_wav_16k_mono() function to standardize WAV files to 16kHz mono format
- Created separate MP3 loader with stereo-to-mono conversion and dynamic resampling
- Applied consistent sampling rate and channel reduction across all inputs

2. Feature Engineering

- Used STFT with 320-sample frames (20ms) and 32-sample hop (90% overlap)
- Truncated/zero-padded all audio to 48,000 samples (3 seconds)
- Automated conversion from raw audio → spectrograms with shape (1491, 257, 1)

3. Dataset Architecture

- Labeled Capuchin calls (1) vs background sounds (0)
- Implemented tf.data with batching(16), shuffling(1000), caching, and prefetching(8)
- 36 batches training, 15 batches testing

4. Cnvolutional Neural Network Model Design

- Architecture: 2×Conv2D(16 filters, 3×3) → Flatten → Dense(128) → Dense(1, sigmoid)
- Training Setup: Adam optimizer, Binary Cross-Entropy loss, Precision/Recall metrics
- Parameters: ~6.1M total parameters, trained for 4 epochs

5. Long Audio Processing

- Split forest recordings into non-overlapping 3-second segments
- Processed multiple windows simultaneously for efficiency
- Handled both WAV training data and MP3 forest recordings

6. Post-Processing & Results

- Applied 0.99 threshold to reduce false positives
- Used itertools.groupby to collapse adjacent detections
- Exported per-file bird call counts to CSV format
