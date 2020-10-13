### Installation

#### Virtual environment setup

Set up your preferred environment and activate it, e.g. do
```bash
python -m venv .venv
source .venv/bin/activate
```
After cloning, simply `cd` into the repo and do `pip install -e .`.

#### Container setup

Using `docker` or `podman`, first build the image with

```bash
docker build -t=costaware -f Dockerfile
```

then run it with your local `costaware` repo mounted in the container using

```bash
docker run -it --rm --privileged -v COSTAWARE_PATH:/home/costaware costaware
```

Any changes made to `/home/costaware` in the container will be reflected
on the host machine in `COSTAWARE_PATH` and vice versa.

### TODOs

- [ ] Get terminal colors working in the container image
- [ ] Write new TODO list
