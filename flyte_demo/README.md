# Flyte Demo

A template for the recommended layout of a Flyte enabled repository for code written in python using [flytekit](https://docs.flyte.org/projects/flytekit/en/latest/).

## Usage

To get up and running with your Flyte project, we recommend following the
[Flyte getting started guide](https://docs.flyte.org/en/latest/getting_started.html).


## Note

1. This APP name is also added to ``docker_build_and_tag.sh`` - ``APP_NAME``
2. We recommend using a git repository and this the ``docker_build_and_tag.sh``
   to build your docker images
3. We also recommend using pip-compile to build your requirements.


## Environment Variables

If you're working on a local sandbox:

```bash
export FLYTE_CONFIG=~/.flyte/config-sandbox.yaml
```

If you're working on a remote cluster, e.g. https://playground.hosted.unionai.cloud

```bash
export FLYTE_CONFIG=~/.flyte/unionplayground-config.yaml
```

Where the contents of `unionplayground-config.yaml` are:

```yaml
admin:
  # For GRPC endpoints you might want to use dns:///flyte.myexample.com
  endpoint: dns:///playground.hosted.unionai.cloud
  authType: Pkce
  # Change insecure flag to ensure that you use the right setting for your environment
  insecure: false
storage:
  type: stow
  stow:
    kind: s3
    config:
      auth_type: iam
      region: us-east-2
logger:
  # Logger settings to control logger output. Useful to debug logger:
  show-source: true
  level: 1
```


## Create Project

```bash
flytectl create project \
   --name flyte-demo \
   --id flyte-demo \
   --description "flyte-demo" \
   --config ~/.flyte/unionplayground-config.yaml \
   --project flyte-demo
```

Update resource attributes:

```
flytectl update task-resource-attribute --attrFile cra.yaml
```


## Building Image

```bash
export VERSION=$(git rev-parse HEAD)

./docker_build_and_tag.sh -a flyte-demo -r ghcr.io/flyteorg -v $VERSION
docker push ghcr.io/flyteorg/flyte-demo:$VERSION

./docker_build_and_tag.sh -a flyte-demo -r ghcr.io/flyteorg -v latest
docker push ghcr.io/flyteorg/flyte-demo:latest
```

## Registering

Package the workflows

```bash
pyflyte --pkgs workflows package --force --image ghcr.io/flyteorg/flyte-demo:latest
```

Register to Backend

```bash
flytectl register files --project flyte-demo --domain development --archive flyte-package.tgz --version $VERSION
```

## Running Workflows

```bash
pyflyte -c $FLYTE_CONFIG run \
   --remote \
   --image ghcr.io/flyteorg/flyte-demo:latest \
   workflows/model_training.py training_workflow \
   --n_epochs 20 \
   --hyperparameters '{"in_dim": 4, "hidden_dim": 100, "out_dim": 3, "learning_rate": 0.03}'
```


## Outline

🚧 NOTE: This demo is still under construction 🚧

This demo covers the following topics

1. Creating tasks and workflows basics
1. The Iteration Cycle: Local to remote execution
1. Scheduling workflows
1. Caching
1. Plugins: BigQuery
1. Custom Types: Dataclasses and Type Extensions
1. Custom Tasks: Datastore
1. Streaming/chunking large data structures
1. CI/CD Testing
1. Gitops Workflow Deployment
