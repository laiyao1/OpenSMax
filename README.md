# OpenSMax

## Introduction
We want to discover botnot attacks by detecting DGA domain names. We propose a method to detect both knwon DGA domain names and 
unknown DGA domain names. This paper is still under reviewing.

## Testing

- **Known Detection**:
Run the run.py file. In this file, people can choose the model, whether split the dataset as class, 
multi classification or two classification, the ratio of class used in the training data.

`python run.py`

-  **Unknown Detection**:
Run the test_unknown_acc.py file.

We support only LSTM(SLD) + One-hot(TLD) now. We will update other models as quickly as possible.

## Requirements
- Python 3.6+
- Tensorflow 
- Keras
- NumPy 1.8+
- Scikit-learn
