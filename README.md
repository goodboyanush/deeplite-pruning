# deeplite-pruning

Take an onnx model and randomly remove x percent (x is a given number between 0 to 100) of conv layers in such a way that the new onnx model is still valid and you can train/test it.

* First select the random conv layers for pruning then remove them one by one (sequentially)
* You may need to adjust the input/output of remaining layers after each layer pruning
* you can test your code on vgg19
* Can you extend your work to support models with residual connections such as resnet family?
* We recommend using mxnet as the AI framework for this coding challenge due to its native support of onnx
    https://mxnet.incubator.apache.org/versions/master/api/python/contrib/onnx.html

# Current Progress

* Working on VGG 19 model
* Performed manual shape inferencing by walking through the computational graph
* Pruned x% of Conv layers and produced a pruned CNN model

# How to Run
The environment works on Python 3.6+ and onnx==1.6.0

1. Install the requirements
```
pip install -r requirements.txt
```

2. Run the main code file for VGG19 pruning
```
python onnx_pruner_vgg19.py
```
Note: The variable `prune_percent` in `__main__` method would define the percentage of pruning

## Console Output

```
Shape inference by walking through the original computational graph
===================================================================
Input Data Shape is:  [3, 224, 224]
After Layer  0 : layer type:  Conv , shape is:  [64, 224, 224]
After Layer  1 : layer type:  Relu , shape is:  [64, 224, 224]
After Layer  2 : layer type:  Conv , shape is:  [64, 224, 224]
After Layer  3 : layer type:  Relu , shape is:  [64, 224, 224]
After Layer  4 : layer type:  MaxPool , shape is:  [64, 112, 112]
After Layer  5 : layer type:  Conv , shape is:  [128, 112, 112]
After Layer  6 : layer type:  Relu , shape is:  [128, 112, 112]
After Layer  7 : layer type:  Conv , shape is:  [128, 112, 112]
After Layer  8 : layer type:  Relu , shape is:  [128, 112, 112]
After Layer  9 : layer type:  MaxPool , shape is:  [128, 56, 56]
After Layer  10 : layer type:  Conv , shape is:  [256, 56, 56]
After Layer  11 : layer type:  Relu , shape is:  [256, 56, 56]
After Layer  12 : layer type:  Conv , shape is:  [256, 56, 56]
After Layer  13 : layer type:  Relu , shape is:  [256, 56, 56]
After Layer  14 : layer type:  Conv , shape is:  [256, 56, 56]
After Layer  15 : layer type:  Relu , shape is:  [256, 56, 56]
After Layer  16 : layer type:  Conv , shape is:  [256, 56, 56]
After Layer  17 : layer type:  Relu , shape is:  [256, 56, 56]
After Layer  18 : layer type:  MaxPool , shape is:  [256, 28, 28]
After Layer  19 : layer type:  Conv , shape is:  [512, 28, 28]
After Layer  20 : layer type:  Relu , shape is:  [512, 28, 28]
After Layer  21 : layer type:  Conv , shape is:  [512, 28, 28]
After Layer  22 : layer type:  Relu , shape is:  [512, 28, 28]
After Layer  23 : layer type:  Conv , shape is:  [512, 28, 28]
After Layer  24 : layer type:  Relu , shape is:  [512, 28, 28]
After Layer  25 : layer type:  Conv , shape is:  [512, 28, 28]
After Layer  26 : layer type:  Relu , shape is:  [512, 28, 28]
After Layer  27 : layer type:  MaxPool , shape is:  [512, 14, 14]
After Layer  28 : layer type:  Conv , shape is:  [512, 14, 14]
After Layer  29 : layer type:  Relu , shape is:  [512, 14, 14]
After Layer  30 : layer type:  Conv , shape is:  [512, 14, 14]
After Layer  31 : layer type:  Relu , shape is:  [512, 14, 14]
After Layer  32 : layer type:  Conv , shape is:  [512, 14, 14]
After Layer  33 : layer type:  Relu , shape is:  [512, 14, 14]
After Layer  34 : layer type:  Conv , shape is:  [512, 14, 14]
After Layer  35 : layer type:  Relu , shape is:  [512, 14, 14]
After Layer  36 : layer type:  MaxPool , shape is:  [512, 7, 7]
After Layer  37 : layer type:  Flatten , shape is:  [1, 25088]
After Layer  38 : layer type:  Gemm , shape is:  [1, 4096]
After Layer  39 : layer type:  Relu , shape is:  [1, 4096]
After Layer  40 : layer type:  Dropout , shape is:  [1, 4096]
After Layer  41 : layer type:  Flatten , shape is:  [1, 4096]
After Layer  42 : layer type:  Gemm , shape is:  [1, 4096]
After Layer  43 : layer type:  Relu , shape is:  [1, 4096]
After Layer  44 : layer type:  Dropout , shape is:  [1, 4096]
After Layer  45 : layer type:  Flatten , shape is:  [1, 4096]
After Layer  46 : layer type:  Gemm , shape is:  [1, 1000]
===================================================================
Is there any error in the recreated model:  None
===================================================================
===================================================================
Saving the model ...
===================================================================
```

## Visualization using Netron
You can find the visualization running in the localhost, port 8080
```
Serving 'vgg19_pruned.onnx' at http://localhost:8080
```

![Netron Plot](images/netron_pruning.png?raw=true "Before and After Pruning")
*Fig. 1: Visualization of the VGG architecture using Netron, before and after pruning*

## Visualization using PyDot
ONNX has an inbuilt visualization using PyDot. The image of the pruned architecture is saved in the following file `pipeline_transpose2x.dot.png`

![Netron Plot](images/pydot_pruning.png?raw=true "Before and After Pruning")

*Fig. 2: Visualization of the VGG architecture using PyDot after pruning*

# Hard Assumptions

* The model is sequential: Currently, support only VGG19 model. If the model is a non-sequential graphical model, then walking through the graph and computing the shapes, require more calculation. Can be done through a stack

* The data is channel first: The shape calculation is computed with the assumption that the data and the shape flow is channel first

* Support limited number of layers: Currently, supporting only `Conv` and `Gemm` as the only trainable layers. `Relu`, `Maxpool`, `Dropout`, `Flatten` as the non-trainable layers.