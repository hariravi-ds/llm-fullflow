import gensim.downloader as api
import torch

model = api.load("word2vec-google-news-300")

word_vectors = model
word_vectors.most_similar(
    positive=["king", "women"], negative=["man"], topn=10)

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
