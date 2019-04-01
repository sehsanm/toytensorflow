from model import *
from layers import *

def calculate_emperical_gradient_simple():
    np.random.seed(0)
    model = Model()
    hl = HiddenLayer(20, 10, 'tanh')
    model.add_layer(hl)
    model.add_layer(BatchSumLayer())


    batch = np.random.randn(40, 20)
    layer_outputs = model.run_batch(batch)
    backward_gradiants = model.backward_run(layer_outputs)

    dw = np.zeros(hl.weights.shape)
    delta = 0.00001
    dw[0, 0] = delta
    hl.apply_backprop(dw)
    layers_outputs_changed = model.run_batch(batch)
    print('Old Model Output  from: ', layer_outputs[-1], ' to ', layers_outputs_changed[-1], ' emperical Gradient: ',
          (layers_outputs_changed[-1] - layer_outputs[-1]) / delta, ' Gradient: ', backward_gradiants[0][1][0, 0])


    graph = GraphModel()
    graph.add_input("input")
    graph.add_node("h", hl, "input")
    graph.add_node("sum" , BatchSumLayer(), "h")
    graph.compile()
    g_output = graph.run_forward(input=batch)
    graph.run_backward(g_output)
    print('Graph Output ',  g_output["sum"].output, ' Gradient:', g_output['h'].gradients[1][0,0])

def calculate_emperical_gradient():
    np.random.seed(0)
    input_dimension = 20
    class_count = 10
    batch_size = 40

    hl = HiddenLayer(input_dimension, class_count, 'tanh')
    batch = np.random.randn(batch_size, input_dimension)
    labels = np.random.rand(batch_size , class_count)

    graph = GraphModel()
    graph.add_input("input")
    graph.add_input("labels")
    graph.add_node("hidden", hl, "input")
    graph.add_node("dummy_softmax" , SoftmaxLayer() , "labels")
    graph.add_node("input_softmax" , SoftmaxLayer() , "hidden")
    graph.add_node("cross_ent" , CrossEntropyLayer() , "dummy_softmax" , "input_softmax" )
    graph.add_node("output" , BatchSumLayer(), "cross_ent")
    graph.compile()
    g_output = graph.run_forward(input=batch, labels=labels)
    graph.run_backward(g_output)
    dw = np.zeros(hl.weights.shape)
    delta = 0.00001
    i = 0
    j = 0
    dw[i, j] = delta
    hl.apply_backprop(dw)

    g2_output = graph.run_forward(input=batch, labels=labels)
    graph.run_backward(g2_output)
    emp_grad = (g2_output["output"].output - g_output["output"].output) / delta
    print('Graph Output ',  g_output["output"].output, ' Gradient:', g_output['hidden'].gradients[1][i,j],
          ' Emperical Grad:' , emp_grad)


if __name__ == "__main__":
    calculate_emperical_gradient_simple()
    calculate_emperical_gradient()
