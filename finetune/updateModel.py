import torch
import tiktoken
from .loadPreTrained import getModel


def updateModel():
    model, BASE_CONFIG = getModel()
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding('gpt2')
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    # shape: (batch_size, num_tokens)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    # shape: (batch_size, num_tokens, num_classes)
    print("Outputs dimensions:", outputs.shape)
    return model
