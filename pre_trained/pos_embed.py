import torch
from in_out_pairs import create_dataloader_v1

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
input, target = first_batch
print("input", input)
print("target", target)

token_embedding_layer = token_embedding_layer(torch.arange(max_length))
positional_embedding = torch.nn.Embedding(max_length, output_dim)

positional_embedding = positional_embedding(torch.arange(max_length))
print(positional_embedding.shape)

positional_embedding = positional_embedding.unsqueeze(0).repeat(8, 1, 1)
final_embeddings = positional_embedding + token_embedding_layer
