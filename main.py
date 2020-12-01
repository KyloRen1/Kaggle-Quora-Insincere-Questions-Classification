import argparse
import pandas as pd

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from text_preprocessing.preprocessing import clean_text_column
from embeddings.utils import load_embedding

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', type=str)
	parser.add_argument('--test_data_path', type=str)
	parser.add_argument('--text_column', type=str)
	parser.add_argument('--target_column', type=str)
	parser.add_argument('--embedding_size', type=int, default=300)
	parser.add_argument('--max_features', type=int, default=120000)
	parser.add_argument('--max_len', type=int, default=70)
	parser.add_argument('--batch_size', type=int, default=512)
	parser.add_argument('--embeddings_folder', type=str)
	return parser.parse_args()

def add_text_features(data, text_col):
	data['question_text'] = data[text_col].progress_apply(lambda x:str(x))
	data['total_length'] = data[text_col].progress_apply(len)
	data['capitals'] = data[text_col].progress_apply(lambda x: sum(1 for c in x if c.isupper()))
	data['caps_vs_length'] = data.progress_apply(lambda row: float(row['capitals']) / float(row['total_length']), axis=1)
	data['num_words'] = data.question_text.str.count('\S+')
	data['num_unique_words'] = data[text_col].progress_apply(lambda x: len(set(w for w in x.split())))
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
	print('===== Finished add text features =====')
	return train, test, train_features, test_features

def tokenize_text(train, test, text_col, target_col, max_features=120000, max_len=70):
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
	print('===== End of text tokenization =====')
	return train_x, train_y, test_x, tokenizer.word_index

if __name__ == '__main__':
	args = get_args()
	train = pd.read_csv(args.train_data_path).head(100)
	test = pd.read_csv(args.test_data_path).head(10)

	print(f'Train data shape: {train.shape}\nTest data shape: {test.shape}')

	train, test = clean_text_column(train, test, args.text_column)
	train, test, train_features, test_features = text_features(train, test, args.text_column)

	train_x, train_y, test_x, word_index = tokenize_text(train, test, args.text_column, args.target_column, args.max_features, args.max_len)
	print(train.head(1))
	print(train_features[0])

	



