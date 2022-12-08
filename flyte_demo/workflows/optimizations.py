from io import BytesIO

import torch
import torch.nn as nn
from flytekit import task, current_context, Resources

from model_training import PenquinsDataset, Hyperparameters


@task(
    # 🎒 Caching
    cache=True,
    cache_version="1",

    # 🔁
    retries=3,

    # 🙏 Request additional resources
    requests=Resources(gpu="1", mem="10Gi"),

    # use spot instances
    interruptible=True,
)
def train_model(
    data: PenquinsDataset, n_epochs: int, hyperparameters: Hyperparameters
) -> nn.Sequential:
    
    # 🚧 Intratask checkpointing:
    # try to get previous checkpoint, if it exists
    try:
        checkpoint = current_context().checkpoint
        prev_checkpoint = checkpoint.read()
    except (NotImplementedError, ValueError):
        checkpoint, prev_checkpoint = None, False

    # assume that checkpoint consists of a counter of the latest epoch and model
    if prev_checkpoint:
        start_epoch, model = torch.load(BytesIO(prev_checkpoint))
    else:
        start_epoch = 0
        model = nn.Sequential(
            nn.Linear(hyperparameters.in_dim, hyperparameters.hidden_dim),
            nn.ReLU(),
            nn.Linear(hyperparameters.hidden_dim, hyperparameters.hidden_dim),
            nn.ReLU(),
            nn.Linear(hyperparameters.hidden_dim, hyperparameters.out_dim),
            nn.Softmax(dim=1),
        )

    for epoch in range(start_epoch, n_epochs):
        # training lop
        ...

        if checkpoint:
            model_io = BytesIO()
            torch.save((epoch, model), model_io)
            model_io.seek(0)
            checkpoint.write(model_io.read())
