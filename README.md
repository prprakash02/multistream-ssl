# README

## Install

```
pip install -r requirements.txt
```

## Run

```
python -m pipeline.train --config configs/default.yaml --device cuda
python -m pipeline.readout --config configs/default.yaml --checkpoint outputs/<run>/best.pt
```

Training writes checkpoints and `metrics.csv` to the run directory. The read-out freezes the encoders, fits
a linear probe and prints the expected offset and influence of every edge.

## Configuration

`configs/default.yaml` carries the model, objective, optimiser and evaluation settings. Streams are declared
under `model.streams`, one entry per stream naming an encoder type and its arguments.

## Data

`pipeline/download/README.md` and `pipeline/preprocess/README.md` give the expected paths and the processed
layout the loader reads.
