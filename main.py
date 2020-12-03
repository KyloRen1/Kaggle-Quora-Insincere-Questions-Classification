import argparse
import pandas as pd
import gc
import random
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from text_preprocessing.preprocessing import clean_text_column
from embeddings.utils import load_embedding
import torch
import time
import os

from models.model import NeuralNetwork


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--text_column', type=str)
    parser.add_argument('--target_column', type=str)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=60)
    parser.add_argument('--linear_hidden_size', type=int, default=16)
    parser.add_argument('--num_capsule', type=int, default=5)
    parser.add_argument('--dim_capsule', type=int, default=5)
    parser.add_argument('--max_features', type=int, default=120000)
    parser.add_argument('--max_len', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embeddings_folder', type=str)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1029)
    return parser.parse_args()


def add_text_features(data, text_col):
    data['question_text'] = data[text_col].progress_apply(lambda x: str(x))
    data['total_length'] = data[text_col].progress_apply(len)
    data['capitals'] = data[text_col].progress_apply(
        lambda x: sum(1 for c in x if c.isupper()))
    data['caps_vs_length'] = data.progress_apply(lambda row: float(
        row['capitals']) / float(row['total_length']), axis=1)
    data['num_words'] = data[text_col].str.count(r'\S+')
    data['num_unique_words'] = data[text_col].progress_apply(
        lambda x: len(set(w for w in x.split())))
    data['words_vs_unique'] = (data['num_unique_words'] / data['num_words'])
    return data


def text_features(train, test, text_col):
    print('===== Text features =====')
    train = add_text_features(train, text_col)
    test = add_text_features(test, text_col)

    train_features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
    test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)

    scaler = StandardScaler()
    scaler.fit(np.vstack((train_features, test_features)))
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    return train, test, train_features, test_features


def tokenize_text(
        train,
        test,
        text_col,
        target_col,
        max_features=120000,
        max_len=70):
    print('===== Text tokenization =====')
    tokenizer = Tokenizer(num_words=max_features, filters='\t\n', lower=True)
    tokenizer.fit_on_texts(list(np.concatenate(
        [train[text_col].values, test[text_col].values])
    ))
    train_x = tokenizer.texts_to_sequences(train[text_col].values)
    test_x = tokenizer.texts_to_sequences(test[text_col].values)

    train_x = pad_sequences(train_x, maxlen=max_len)
    test_x = pad_sequences(test_x, maxlen=max_len)
    train_y = train[target_col].values
    return train_x, train_y, test_x, tokenizer.word_index


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate(model, criterion, val_loader):
    avg_eval_loss = 0.
    eval_predictions = list()
    model.eval()
    for i, (batch_x, batch_y) in enumerate(val_loader):
        features = kfold_x_features[i * batch_size: (i + 1) * batch_size]
        y_predictions = model([batch_x, features])

        loss = criterion(y_predictions, batch_y)
        avg_eval_loss += loss.item()
        eval_predictions.append(
            sigmoid(
                y_predictions.detach().cpu().numpy())[
                :, 0])

    return avg_eval_loss / len(val_loader), eval_predictions


def train(model, criterion, optimizer, scheduler, train_loader):
    avg_train_loss = 0.
    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        features = kfold_x_features[i * batch_size: (i + 1) * batch_size]
        y_predictions = model([batch_x, features])

        scheduler.batch_step()
        loss = criterion(y_predictions, batch_y)
        optmizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.item()

    return model, avg_trian_loss / len(train_loader)


def inference(model, test_loader):
    test_predictions = list()
    model.eval()
    for i, (batch_x, ) in enumerate(test_loader):
        features = test_features[i * batch_size: (i + 1) * batch_size]
        y_predictions = model([batch_x, features])
        test_predictions.append(
            sigmoid(
                y_predictions.detach().cpu().numpy())[
                :, 0])
    return


def train_fold():
    for epoch in range(epochs):
        start_time = time.time()

        model, train_loss = train(
            model, criterion, optimizer, scheduler, train_loader)
        val_loss, val_preds = evaluate(model, criterion, val_loader)


if __name__ == '__main__':
    args = get_args()
    train = pd.read_csv(args.train_data_path).head(100)
    test = pd.read_csv(args.test_data_path).head(10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Train data shape: {train.shape}\nTest data shape: {test.shape}')

    train, test = clean_text_column(train, test, args.text_column)
    train, test, train_features, test_features = text_features(
        train, test, args.text_column)

    train_x, train_y, test_x, word_index = tokenize_text(
        train, test, args.text_column, args.target_column, args.max_features, args.max_len)

    print('===== Loading embeddings =====')
    glove_embedding = load_embedding(
        word_index,
        'glove',
        args.embeddings_folder,
        args.max_features)
    paragram_embedding = load_embedding(
        word_index,
        'paragram',
        args.embeddings_folder,
        args.max_features)
    embedding_matrix = np.mean([glove_embedding, paragram_embedding], axis=0)
    del glove_embedding, paragram_embedding
    gc.collect()
    print('Embedding shape: ', embedding_matrix.shape)

    print('==== Creating folds =====')
    splits = list(StratifiedKFold(
        n_splits=args.num_splits, shuffle=True, random_state=args.seed
    ).split(train_x, train_y))

    train_preds = np.zeros((len(train_x)))
    test_preds = np.zeros((len(test_x)))

    seed_everything()

    model = NeuralNetwork(
        args.max_features,
        args.embedding_size,
        embedding_matrix,
        args.hidden_size,
        args.linear_hidden_size,
        args.num_capsule,
        args.dim_capsule)
