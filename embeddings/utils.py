import numpy as np
from tqdm import tqdm

def load_embedding(word_index, embedding_name, embeddings_folder, max_features=120000):
	if embedding_name == 'glove':
		EMBEDDING_FILE = embeddings_folder+'/glove.840B.300d/glove.840B.300d.txt'
	elif embedding_name == 'wikinews':
		EMBEDDING_FILE = embeddings_folder+'/wiki-news-300d-1M/wiki-news-300d-1M.vec'
	else:
		EMBEDDING_FILE = embeddings_folder+'/paragram_300_sl999/paragram_300_sl999.txt'
	def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

	embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE, encoding='utf8', errors='ignore') if len(o) > 100)
	all_embeddings = np.stack(embeddings_index.values())
	embedding_mean, embedding_std = all_embeddings.mean(), all_embeddings.std()
	embedding_size = all_embeddings.shape[1]


	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.random.normal(embedding_mean, embedding_std, (nb_words, embedding_size))
	for word, i in tqdm(word_index.items(), desc=f'loading {embedding_name} embedding'):
		if i >= nb_words: continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			embedding_vector = embeddings_index.get(word.upper())
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
			else:
				embedding_vector = embeddings_index.get(word.capitalize())
				if embedding_vector is not None:
					embedding_matrix[i] = embedding_vector
	return embedding_matrix