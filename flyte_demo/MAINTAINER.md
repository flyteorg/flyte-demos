# For Maintainers

This document is for maintainers of this repo.

## Building Image

To expose a pre-built docker image for users, build and push to the github
docker container registry:

```bash
export VERSION=$(git rev-parse HEAD)

# build and push a docker image with git sha as the version
./docker_build_and_tag.sh -a flyte-demo -r ghcr.io/flyteorg -v $VERSION
docker push ghcr.io/flyteorg/flyte-demo:$VERSION

# (optional) build and push with the latest tag
./docker_build_and_tag.sh -a flyte-demo -r ghcr.io/flyteorg -v latest
docker push ghcr.io/flyteorg/flyte-demo:latest
```
