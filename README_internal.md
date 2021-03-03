# Key changes

1. In `dataloader/train_loader.py`, uncomment the following line 100,
we don't merge the cell classes for SCRC

```
# type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
```

2. Replace the original `type_info.json` with the version of SCRC

3. In `config.py` we use the following configuration, which is bascially the same to the original config.

```
model_mode = "fast"
...
act_shape = [256, 256]
out_shape = [164, 164]  # patch shape at output of network
```

4. Replace `train` and `val` dirs with input arguments for `run_train.py`, originally this is passed as lists defined in `config.py`

```
...
self.train_dir = train_dir
self.valid_dir = valid_dir
...
if run_mode == "train":
    data_dir = self.train_dir
else:
    data_dir = self.valid_dir
```