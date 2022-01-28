
#### checkpoints.json
Should contain the model folder name, the model dimensions and an index for each of the model. The indices are used to refer to the ensemble weights in the deep-chimpact.yaml config file. So the overall structure is given below:
```
[
    ["Model Folder Name", [model_dims[0],model_dims[1]],model_index],
    ...
]
```

#### deep-chimpact.yaml
Config file for training and infer, contains the necessary parameters for training. 

