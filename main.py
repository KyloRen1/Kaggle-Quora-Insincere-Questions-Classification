import os
import gc
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from text_preprocessing.preprocessing import clean_text_column
from text_preprocessing.features import text_features, tokenize_text
from embeddings.utils import load_embedding
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
    parser.add_argument('--learning_rate_max', type=float, default=0.003)
    parser.add_argument('--learning_rate_base', type=float, default=0.001)
    parser.add_argument('--scheduler_step_size', type=int, default=300)
    parser.add_argument('--scheduler_gamma', type=float, default=0.99994)
    return parser.parse_args()


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(
        splits,
        train_x,
        train_y,
        test_x,
        features,
        test_features,
        embedding_matrix,
        args,
        device):

    test_x = torch.tensor(test_x, dtype=torch.long).to(device)
    test = torch.utils.data.TensorDataset(test_x)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=args.batch_size, shuffle=False)

    train_loss_history = list()
    val_loss_history = list()

    train_preds = np.zeros((len(train_x)))
    test_preds = np.zeros((len(test_x)))

    for i, (train_idx, val_idx) in enumerate(splits):
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        features = np.array(features)

        train_x_fold = torch.tensor(
            train_x[train_idx.astype(int)], dtype=torch.long).to(device)
        train_y_fold = torch.tensor(train_y[train_idx.astype(
            int), np.newaxis], dtype=torch.float32).to(device)
        val_x_fold = torch.tensor(
            train_x[val_idx.astype(int)], dtype=torch.long).to(device)
        val_y_fold = torch.tensor(train_y[val_idx.astype(
            int), np.newaxis], dtype=torch.float32).to(device)
        kfold_x_train_features = features[train_idx.astype(int)]
        kfold_x_val_features = features[val_idx.astype(int)]

        model = NeuralNetwork(
            args.max_features,
            args.embedding_size,
            embedding_matrix,
            args.hidden_size,
            args.linear_hidden_size,
            args.num_capsule,
            args.dim_capsule,
            device).to(device)

        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                model.parameters()),
            lr=args.learning_rate_max)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.learning_rate_base,
            max_lr=args.learning_rate_max,
            mode='exp_range',
            gamma=args.scheduler_gamma,
            cycle_momentum=False)

        train = torch.utils.data.TensorDataset(train_x_fold, train_y_fold)
        val = torch.utils.data.TensorDataset(val_x_fold, val_y_fold)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=args.batch_size, shuffle=False)

        print(f' ===== Fold {i + 1} ===== ')
        for epoch in range(args.num_epochs):
            start_time = time.time()

            model.train()
            avg_train_loss = 0.

            for i, (batch_x, batch_y) in enumerate(train_loader):
                feats = kfold_x_train_features[i * \
                    args.batch_size: (i + 1) * args.batch_size]
                y_predictions = model([batch_x, feats])

                loss = criterion(y_predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                avg_train_loss += loss.item()

            avg_train_loss = avg_train_loss / len(train_loader)

            model.eval()

            val_preds_fold = np.zeros((val_x_fold.size(0)))
            test_preds_fold = np.zeros((test_x.size(0)))

            avg_val_loss = 0.
            for i, (batch_x, batch_y) in enumerate(val_loader):
                feats = kfold_x_val_features[i * \
                    args.batch_size: (i + 1) * args.batch_size]
                y_predictions = model([batch_x, feats])

                loss = criterion(y_predictions, batch_y)
                avg_val_loss += loss.item()
                val_preds_fold[i * args.batch_size: (i + 1) * args.batch_size] = sigmoid(
                    y_predictions.detach().cpu().numpy())[:, 0]
            avg_val_loss = avg_val_loss / len(val_loader)

            print(
                'EPOCH {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1,
                    args.num_epochs,
                    avg_train_loss,
                    avg_val_loss,
                    time.time() - start_time))

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)

        for i, (batch_x, ) in enumerate(test_loader):
            feats = test_features[i *
                                  args.batch_size: (i + 1) * args.batch_size]
            y_predictions = model([batch_x, feats])
            test_preds_fold[i * args.batch_size: (i + 1) * args.batch_size] = sigmoid(
                y_predictions.detach().cpu().numpy())[:, 0]

        train_preds[val_idx] = val_preds_fold
        test_preds += test_preds_fold / len(splits)

    print('Total loss={:.4f} \t val_loss={:.4f} \t'.format(
        np.average(train_loss_history), np.average(val_loss_history)))
    return train_preds, test_preds


def best_threshold(y_train, train_preds):
    tmp = [0, 0, 0]
    delta = 0
    for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
        tmp[1] = f1_score(y_train, np.array(train_preds) > tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    print(
        'best threshold is {:.4f} with F1 score: {:.4f}'.format(
            delta,
            tmp[2]))
    return delta


if __name__ == '__main__':
    args = get_args()
    train = pd.read_csv(args.train_data_path).head(1000)
    test = pd.read_csv(args.test_data_path).head(100)
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

    seed_everything()

    print('===== Training a model =====')
    train_preds, test_preds = train_model(
        splits, train_x, train_y, test_x, train_features, test_features, embedding_matrix, args, device)

    delta = best_threshold(train_y, train_preds)

    submission = test[['qid']].copy()
    submission['prediction'] = (test_preds > delta).astype(int)
    submission.to_csv('submission.csv', index=False)
    print('===== Finished training =====')
