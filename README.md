# Kaggle: Quora Insincere Questions Classification.

This is the 59th place solution for the Kaggle competition Quora: Insincere Questions Classification.

## The Problem
This was a binary classification problem where we were required to predict whether the question contains abusive content or not. More info here: https://www.kaggle.com/c/quora-insincere-questions-classification

## Model 
The code of the model is located in `models\model.py`.
The model architecture consists of a few layers:
I have averaged GloVe and PARAGRAM embeddings and loaded this vector to the `torch.nn.Embedding` layer. The concatenation of `LSTM with Attention`, `GRU with Attention`, `GRU with MaxPooling`, `GRU with AveragePooling`, and `Capsule` layer (taken from here https://github.com/binzhouchn/capsule-pytorch) over GRU layer. Head output is composed of the `Linear` layer with `ReLU` activation and a `Linear` layer. 

This model reaches an F1-score of 0.69764 in Public and 0.70489 in Private leaderboards

## Code
To test the algorithm place `train.csv`, `test.csv`, `embeddings` in folder `data`. Afterwards run the following code:

`python3 main.py --train_data_path='data/train.csv' --test_data_path='data/test.csv' --text_column='question_text' --target_column='target' --embeddings_folder='data/embeddings'`
