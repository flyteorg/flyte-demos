"""Flyte Intro: Tasks and Workflows.

These examples will use the penguins dataset:
https://allisonhorst.github.io/palmerpenguins/

Using the pypi package:
https://pypi.org/project/palmerpenguins/
"""

from dataclasses import dataclass
from typing import Annotated, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from palmerpenguins import load_penguins
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataclasses_json import dataclass_json
from flytekit import task, workflow, kwtypes, Resources, LaunchPlan, CronSchedule
from flytekit.types.structured.structured_dataset import StructuredDataset


TARGET = "species"
CLASSES = ["Adelie", "Chinstrap", "Gentoo"]
FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

@dataclass_json
@dataclass
class Hyperparameters:
    in_dim: int
    hidden_dim: int
    out_dim: int
    learning_rate: float


PenquinsDataset = Annotated[
    StructuredDataset,
    kwtypes(
        species=str,
        bill_length_mm=float,
        bill_depth_mm=float,
        flipper_length_mm=float,
        body_mass_g=float,
    ),
]


@task
def get_data() -> pd.DataFrame:
    return load_penguins()[[TARGET] + FEATURES].dropna()


@task
def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        data, test_size=test_size, random_state=random_state
    )


@task
def preprocess_splits(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mean = train_data[FEATURES].mean()
    std = train_data[FEATURES].std()

    train_data[FEATURES] = (train_data[FEATURES] - mean) / std
    test_data[FEATURES] = (test_data[FEATURES] - mean) / std
    return train_data, test_data


@task(requests=Resources(cpu="2", mem="1Gi"))
def train_model(
    data: PenquinsDataset, n_epochs: int, hyperparameters: Hyperparameters
) -> nn.Sequential:
    # extract features and targets
    data = data.open(pd.DataFrame).all()
    features = torch.from_numpy(data[FEATURES].values).float()
    targets = torch.from_numpy(pd.get_dummies(data[TARGET])[CLASSES].values).float()

    # create model
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

    # train for n_epochs
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(features), targets)
        print(f"loss={loss:.04f}")
        loss.backward()
        opt.step()

    return model


@task
@torch.no_grad()
def evaluate(model: nn.Sequential, data: pd.DataFrame) -> float:
    features = torch.from_numpy(data[FEATURES].values).float()
    pred = [CLASSES[i] for i in model(features).argmax(dim=1).numpy()]
    return float(accuracy_score(data[TARGET], pred))


@workflow
def training_workflow(
    n_epochs: int,
    hyperparameters: Hyperparameters,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[nn.Sequential, float, float]:
    # get and split data
    data = get_data()
    train_data, test_data = split_data(
        data=data, test_size=test_size, random_state=random_state
    )
    train_data, test_data = preprocess_splits(
        train_data=train_data, test_data=test_data,
    )

    # train model on the training set
    model = train_model(
        data=train_data, n_epochs=n_epochs, hyperparameters=hyperparameters
    )

    # evaluate the model
    train_acc = evaluate(model=model, data=train_data)
    test_acc = evaluate(model=model, data=test_data)

    # return model with results
    return model, train_acc, test_acc


training_launchplan = LaunchPlan.create(
    "scheduled_training_workflow",
    training_workflow,

    # run 2 minutes
    schedule=CronSchedule(schedule="*/3 * * * *"),

    # use default inputs
    default_inputs={
        "n_epochs": 30,
        "hyperparameters": Hyperparameters(
            in_dim=4, hidden_dim=100, out_dim=3, learning_rate=0.03
        ),
    },
)


if __name__ == "__main__":
    # You can run workflows locally, it's just Python 🐍!
    hyperparameters = Hyperparameters(
        in_dim=4, hidden_dim=100, out_dim=3, learning_rate=0.01
    )
    print(f"{training_workflow(n_epochs=30, hyperparameters=hyperparameters)}")
