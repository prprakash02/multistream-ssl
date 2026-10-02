# Preprocessing

Place your preprocessing code here, one module per source. It resamples every stream of a clip onto a
common grid of `T` frames and writes the layout the loader reads:

```
data/processed/<name>/
  meta.json                 # {"T": int, "num_classes": int, "streams": {name: {"type": ..., ...}}}
  manifest_{train,val,test}.csv   # columns: id, path, label
  samples/<id>.npz          # arr_<stream> of shape [T, *raw] per stream, plus label
```

The stream names and encoder types in `meta.json` must match `model.streams` in the configuration.
