import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc = tiktoken.get_encoding('gpt2')
enc_text = enc.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
context_size = 4

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)


class GPTDatasetV1(Dataset):
    def __init__(self, text, enc, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = enc.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length + 1, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # batch size --> number of batches the model has to run before updating its parameters
    # stride prevents over fitting
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
