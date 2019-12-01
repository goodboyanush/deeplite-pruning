"""
    Deeplite coding challenge: onnx_pruner.
	The main goal of this coding challenge is to implement a method to prune conv layers of a given onnx model.

	Details:
    Take an onnx model and randomly remove x percent (x is a given number between 0 to 100) of conv layers in such
    a way that the new onnx model is still valid and you can train/test it.

	** First select the random conv layers for pruning then remove them one by one (sequentially)
	** You may need to adjust the input/output of remaining layers after each layer pruning
    ** you can test your code on vgg19
    ** Can you extend your work to support models with residual connections such as resnet family?
    ** We recommend using mxnet as the AI framework for this coding challenge due to its native support of onnx
       https://mxnet.incubator.apache.org/versions/master/api/python/contrib/onnx.html
"""

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import helper, shape_inference
import copy, math

def number_of_layers_to_prune(model, x, layer_type='Conv'):
    """
    :param model: onnx model
    :param x: pruning ratio (0 to 100)
    :param layer_type: layer to be pruned ('Conv')
    :return: prune count
    """
    conv_count = 0
    for i in range(len(onnx_model.graph.node)):
        if layer_type in onnx_model.graph.node[i].op_type:
            conv_count += 1
            
    prune_count = math.floor(conv_count * (prune_percent/ 100))
    return prune_count

def calc_dimension_change(size, op_type, node_params, node_inputs):
    """
    :param size: current running size (n-dimesnional tuple)
    :param op_type: current operation being performed
    :param node_params: parameters/attributes of the node being operated
    :param node_input: input shapes to the current node
    :return: updated new size
    """
    newSize = [0,0,0]
    if (op_type == "Conv"):
        params = {'dilations_row': 1, 'dilations_col': 1, 'kernel_shape_row': 3, 'kernel_shape_col': 3, 
                  'pads_row': 0, 'pads_col': 0, 'strides_row': 1, 'strides_col': 1, "nb_filter": 64}
        for i in node_inputs:
            if i.name == node_params.input[1]:
                nb_filter = i.type.tensor_type.shape.dim[0].dim_value
        for i in node_params.attribute:
            if (i.name == 'dilations'):
                params['dilations_row'] = i.ints[0]
                params['dilations_col'] = i.ints[1]
            elif (i.name == 'kernel_shape'):
                params['kernel_shape_row'] = i.ints[0]
                params['kernel_shape_col'] = i.ints[1]
            elif (i.name == 'pads'): # for now assume, that both start and end dimensions for each axis is the same
                params['pads_row'] = i.ints[0]
                params['pads_col'] = i.ints[2]
            elif (i.name == 'strides'):
                params['strides_row'] = i.ints[0]
                params['strides_col'] = i.ints[1]
    
        newSize[1] = int((size[1] - params['kernel_shape_row'] + 2 * params['pads_row']) / params['strides_row'] + 1);
        newSize[2] = int((size[2] - params['kernel_shape_col'] + 2 * params['pads_col']) / params['strides_col'] + 1);
        newSize[0] = nb_filter;
    
    if (op_type == "MaxPool"):
        params = {'dilations_row': 1, 'dilations_col': 1, 'kernel_shape_row': 3, 'kernel_shape_col': 3, 
                  'pads_row': 0, 'pads_col': 0, 'strides_row': 1, 'strides_col': 1, "nb_filter": 64}
        for i in node_params.attribute:
            if (i.name == 'dilations'):
                params['dilations_row'] = i.ints[0]
                params['dilations_col'] = i.ints[1]
            elif (i.name == 'kernel_shape'):
                params['kernel_shape_row'] = i.ints[0]
                params['kernel_shape_col'] = i.ints[1]
            elif (i.name == 'pads'): # for now assume, that both start and end dimensions for each axis is the same
                params['pads_row'] = i.ints[0]
                params['pads_col'] = i.ints[2]
            elif (i.name == 'strides'):
                params['strides_row'] = i.ints[0]
                params['strides_col'] = i.ints[1]
    
        newSize[1] = int((size[1] - params['kernel_shape_row'] + 2 * params['pads_row']) / params['strides_row'] + 1);
        newSize[2] = int((size[2] - params['kernel_shape_col'] + 2 * params['pads_col']) / params['strides_col'] + 1);
        newSize[0] = size[0];
    
    if (op_type == "Gemm"):
        nb_nodes = 0
        for i in node_inputs:
            if i.name == node_params.input[1]:
                nb_nodes = int(i.type.tensor_type.shape.dim[0].dim_value)
        newSize = [1, nb_nodes]
        
    if (op_type == "Flatten"):
        totalSize = 1
        for i in size:
            totalSize = totalSize * i
        newSize = [1, totalSize]
    
    if (op_type == "Relu"):
        newSize = size
    
    if (op_type == "Dropout"):
        newSize = size
        
    return newSize

def prune(model, x):
    """
    :param model: onnx model
    :param x: pruning ratio (0 to 100)
    :return: pruned model
    """
    pass

    prune_count = number_of_layers_to_prune(model, x, 'Conv')

    # Assuming the shape is channel first
    current_shape = [onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value, 
                    onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value, 
                    onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value]

    nodes = []
    all_inputs = []
    flag = True
    number_pruned = 0
    print("Shape inference by walking through the original computational graph")
    print("===================================================================")
    print("Input Data Shape is: ", current_shape)
    for i in range(len(onnx_model.graph.node)):
        current_shape = calc_dimension_change(current_shape, onnx_model.graph.node[i].op_type, onnx_model.graph.node[i], onnx_model.graph.input)
        print("After Layer ", i, ": layer type: ", onnx_model.graph.node[i].op_type, ", shape is: ", current_shape)
        # If the current layer is Conv Layer, then the next layer can be skipped as it is already added
        if not flag:
            flag = True
            continue
        # If the current layer is Conv Layer, skip the connection and add it to the next layer
        if 'Conv' in onnx_model.graph.node[i].op_type and number_pruned <= prune_count:
            current_node = copy.copy(onnx_model.graph.node[i+1])
            # create a new node
            if i > 0:
                current_node.input[0] = onnx_model.graph.node[i-1].output[0]
            else:
                current_node.input[0] = 'data'
                all_inputs.append(onnx_model.graph.input[0])
                
            nodes.append(current_node)
            flag = False
            number_pruned += 1
        # Any layer apart from Conv Layer
        else:
            # Get all the input parameters of this layer
            current_layer_inputs = []
            for curr_input in onnx_model.graph.node[i].input:
                current_layer_inputs.append(curr_input)
                if curr_input != nodes[-1].output[0]:
                    for k, graph_input in enumerate(onnx_model.graph.input):
                        if graph_input.name == curr_input:
                            all_inputs.append(onnx_model.graph.input[k])
            
            # Recreate Learnable layers only to add the input parameters
            # FC/ Dense/ GEMM (General Matrix Multiplication) Layer
            if "Gemm" in onnx_model.graph.node[i].op_type:   
                new_node = helper.make_node(onnx_model.graph.node[i].op_type, 
                                            inputs = current_layer_inputs, 
                                            outputs = onnx_model.graph.node[i].output,
                                            alpha=1.0,
                                            beta=1.0,
                                            transA=0,
                                            transB=1)    
                
                nodes.append(new_node)
            # Convolution layer
            elif "Conv" in onnx_model.graph.node[i].op_type:
                new_node = helper.make_node(onnx_model.graph.node[i].op_type, 
                                            inputs = current_layer_inputs, 
                                            outputs = onnx_model.graph.node[i].output)   
                nodes.append(new_node)
            # If the layer is non-learnable, add it as such
            else:
                nodes.append(onnx_model.graph.node[i])

    graph_def = helper.make_graph(
        nodes,
        'test-model',
        all_inputs,
        onnx_model.graph.output
    )

    model_def = helper.make_model(graph_def, producer_name='test-model')

    inferred_model = shape_inference.infer_shapes(model_def)
    print("===================================================================")
    print("Is there any error in the recreated model: ", onnx.checker.check_model(inferred_model))
    print("===================================================================")
    return inferred_model

if __name__ == '__main__':
    onnx_model = onnx.load('vgg19.onnx')
    prune_percent = 50

    inferred_model = prune(onnx_model, prune_percent)

    print("===================================================================")
    print("Saving the model ... ")
    onnx.save(inferred_model, 'vgg19_pruned.onnx')
    print("===================================================================")

    ########################## Additional functions, hoping to impress :) #######################
    # If you want to visualize the ONNX model using NETRON
    visualize_NETRON = True
    if visualize_NETRON:
        import netron
        netron.start('vgg19_pruned.onnx', False, False, 8080)

    # If you want to visualize the ONNX model using PyDot
    from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
    pydot_graph = GetPydotGraph(inferred_model.graph, name=inferred_model.graph.name, rankdir="TB",
                                node_producer=GetOpNodeProducer("docstring", color="yellow",
                                                                fillcolor="yellow", style="filled"))
    pydot_graph.write_dot("pipeline_transpose2x.dot")

    import os
    os.system('dot -O -Gdpi=500 -Tpng pipeline_transpose2x.dot')

    import matplotlib.pyplot as plt
    image = plt.imread("pipeline_transpose2x.dot.png")
    fig, ax = plt.subplots(figsize=(400, 200))
    ax.imshow(image)
    ax.axis('off')
    
