# Flyte Demo

A template for the recommended layout of a Flyte enabled repository for code written in python using [flytekit](https://docs.flyte.org/projects/flytekit/en/latest/).

## Setup

First, install [flytectl](https://docs.flyte.org/projects/flytectl/en/latest/#installation).

Then start a local Flyte cluster:

```
flytectl demo start --source .
```

Expected output:

```
👨‍💻 Flyte is ready! Flyte UI is available at http://localhost:30080/console 🚀 🚀 🎉
...
```

Make sure to export the `KUBECONFIG` environment variable that's output after
the previous command has completed.

**Important**: Set `FLYTECTL_CONFIG` to `./config-sandbox.yaml`.

```
export FLYTECTL_CONFIG=./config-sandbox.yaml
```

Update the default resources available to tasks in your local cluster:

```
flytectl update task-resource-attribute --attrFile cra.yaml
```

## Setting a Docker Image

This repo ships with pre-built docker images, which can be found
[here](https://github.com/flyteorg/flyte-demos/pkgs/container/flyte-demo).

Use this image for the rest of this README guide.

```
export IMAGE=ghcr.io/flyteorg/flyte-demo:latest
```

**[Optional]** Alternatively, if you want to build the image inside the local
Flyte cluster, you can do so with the following command:

```
flytectl demo exec -- docker build . --tag "flyte-demo:v1"
export IMAGE=flyte-demo:v1
```

## Project Structure

This repo contains a `workflows` folder, which contains the following workflow
modules:

- `model_training.py`: a simple model training example
- `optimizations.py`: examples of how to optimize your training code with caching,
  retries, resource requests, and intratask checkpointing.
- `data_iter.py`: an example of a custom parquet encode/decoder for loading
  chunks of data into a Flyte etask.
- `bigquery.py`: using `flytekitplugins-bigquery` to read data from a BigQuery DB.
- `visualizations.py`: examples of Flyte deck extensions for custom visualizations.

## Registering

Package the workflows

```bash
pyflyte --pkgs workflows package --force --image $IMAGE
```

Register to Backend

```bash
flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version v1
```

## Running Workflows

```bash
pyflyte run \
   --remote \
   --image $IMAGE \
   workflows/model_training.py training_workflow \
   --n_epochs 20 \
   --hyperparameters '{"in_dim": 4, "hidden_dim": 100, "out_dim": 3, "learning_rate": 0.03}'
```

To run these workflows on a remote Flyte cluster, follow the **Remote Flyte Cluster**
instructions in [these guides](https://docs.flyte.org/projects/cookbook/en/latest/auto/larger_apps/index.html).

## Notebook Demo

Follow the `demo.ipynb` notebook to run workflows and interact with a Flyte
cluster from a Jupyter notebook runtime.


## Script Demo

The `run_model_training.py` script provides an example of how to use the
`FlyteRemote` object to run workflows from a Python script.
