import torch.nn as nn

from flytekit.configuration import Config
from flytekit.remote import FlyteRemote
from workflows import model_training

remote = FlyteRemote(
    config=Config.auto(),
    default_project="flytesnacks",
    default_domain="development",
)
execution = remote.execute_local_workflow(
    model_training.training_workflow,
    inputs={
        "n_epochs": 30,
        "hyperparameters": model_training.Hyperparameters(
            in_dim=4, hidden_dim=100, out_dim=3, learning_rate=0.01
        ),
    }
)

print(remote.generate_console_url(execution))
execution = remote.wait(execution)
print(execution.outputs.get("o0", as_type=nn.Sequential))
