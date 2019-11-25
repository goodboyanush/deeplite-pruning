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

def prune(model, x):
    """
    :param model: onnx model
    :param x: pruning ratio (0 to 100)
    :return: pruned model
    """
    pass

    prune_count = number_of_layers_to_prune(model, x, 'Conv')

    nodes = []
    all_inputs = []
    flag = True
    number_pruned = 0
    for i in range(len(onnx_model.graph.node)):
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

    # Create an ONNX Graph
    graph_def = helper.make_graph(
        nodes,
        'test-model',
        all_inputs,
        onnx_model.graph.output
    )

    # Create an ONNX Model
    model_def = helper.make_model(graph_def, producer_name='test-model')

    # Perform Shape inference
    try:
        inferred_model = shape_inference.infer_shapes(model_def)
    except:
        print("Shape inference failed")

    # Check the newly created ONNX model
    print("Errors in the created ONNX model: ", onnx.checker.check_model(onnx_model))

    return inferred_model

if __name__ == '__main__':
    onnx_model = onnx.load('vgg19.onnx')
    prune_percent = 50

    inferred_model = prune(onnx_model, prune_percent)

    onnx.save(inferred_model, 'vgg19_pruned.onnx')

    ########################## Additional functions, hoping to impress :) #######################
    # If you want to visualize the ONNX model using NETRON
    visualize_NETRON = True
    if visualize_NETRON:
        import netron
        netron.start('vgg19_pruned.onnx', False, False, 8080)

    
