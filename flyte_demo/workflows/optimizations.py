from io import BytesIO
from random import random
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flytekit import current_context, task, workflow, Resources
from flytekit.exceptions.user import FlyteRecoverableException

from workflows.model_training import (
    PenquinsDataset,
    Hyperparameters,
    FEATURES,
    TARGET,
    CLASSES,
    get_data,
    split_data,
    preprocess_splits,
    evaluate,
)


@task(
    # 🎒 Caching
    cache=True,
    cache_version="1",

    # 🔁 Retry on error
    retries=3,

    # 🙏 Request additional resources
    # Set gpu="1" to use gpus
    requests=Resources(cpu="1", mem="1Gi"),

    # Set this to True to use spot instances
    interruptible=False,
)
def train_model(
    data: PenquinsDataset, n_epochs: int, hyperparameters: Hyperparameters
) -> nn.Sequential:

    # extract features and targets
    data = data.open(pd.DataFrame).all()
    features = torch.from_numpy(data[FEATURES].values).float()
    targets = torch.from_numpy(pd.get_dummies(data[TARGET])[CLASSES].values).float()
    
    # 🚧 Intratask checkpointing:
    # try to get previous checkpoint, if it exists
    try:
        checkpoint = current_context().checkpoint
        prev_checkpoint = checkpoint.read()
    except (NotImplementedError, ValueError):
        checkpoint, prev_checkpoint = None, False

    # assume that checkpoint consists of a counter of the latest epoch and model
    if prev_checkpoint:
        start_epoch, model, opt = torch.load(BytesIO(prev_checkpoint))
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
        opt = torch.optim.Adam(
            model.parameters(), lr=hyperparameters.learning_rate
        )

    for epoch in range(start_epoch, n_epochs):

        # simulate system-level error: per epoch, introduce
        # a chance of failure 5% of the time
        if random() < 0.05:
            raise FlyteRecoverableException(
                f"🔥 Something went wrong at epoch {epoch}! 🔥"
            )

        # training loop
        opt.zero_grad()
        loss = F.cross_entropy(model(features), targets)
        print(f"loss={loss:.04f}")
        loss.backward()
        opt.step()

        if checkpoint:
            model_io = BytesIO()
            torch.save((epoch, model, opt), model_io)
            model_io.seek(0)
            checkpoint.write(model_io.read())


@workflow
def training_workflow(
    n_epochs: int,
    hyperparameters: Hyperparameters,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[nn.Sequential, float, float]:
    data = get_data()
    train_data, test_data = split_data(data=data, test_size=test_size, random_state=random_state)
    train_data, test_data = preprocess_splits(train_data=train_data, test_data=test_data)
    model = train_model(data=train_data, n_epochs=n_epochs, hyperparameters=hyperparameters)
    train_acc = evaluate(model=model, data=train_data)
    test_acc = evaluate(model=model, data=test_data)
    return model, train_acc, test_acc
