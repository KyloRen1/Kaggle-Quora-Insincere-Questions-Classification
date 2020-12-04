# Kaggle: Quora Insincere Questions Classification
This is the 59th place solution for Kaggle competition Quora: Insincere Questions Classification

## The Problem
This was a binary classification problem where we were required to predict whether the question contains abusive content or not. More info here: https://www.kaggle.com/c/quora-insincere-questions-classification

## Model and training

__Preprocessing step__ <br\>
&nbsp; 1. __Text preprocessing__: lowercase the text, remove punctuation, remove numbers, remove typical misspellings
&nbsp; 2. __New features__: ratio of capital letters, ration of unique words

### Model
I have averaged GloVe and PARAGRAM embeddings and loaded this vector to `torch.nn.Embedding` layer. 



## Code
To test the algorithm place `train.csv`, `test.csv`, `embeddings` in folder `data`. Afterwards run the following code:

`python3 main.py --train_data_path='data/train.csv' --test_data_path='data/test.csv' --text_column='question_text' --target_column='target' --embeddings_folder='data/embeddings'`
