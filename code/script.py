import datetime
import os
import shutil

from model import *
from layers import *
import pickle
import matplotlib.pyplot as plt



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
    g_output = graph.run_forward({'input':batch})
    graph.run_backward(g_output)
    print('Graph Output ',  g_output["sum"].output, ' Gradient:', g_output['h'].gradients[1][0,0])


def calculate_emperical_gradient():
    np.random.seed(0)
    input_dimension = 20
    class_count = 10
    batch_size = 400

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
    g_output = graph.run_forward({'input':batch, 'labels':labels})
    graph.run_backward(g_output)
    dw = np.zeros(hl.weights.shape)
    delta = 0.00001
    i = 0
    j = 0
    dw[i, j] = delta
    hl.apply_backprop(dw)

    g2_output = graph.run_forward({'input':batch, 'labels':labels})
    graph.run_backward(g2_output)
    emp_grad = (g2_output["output"].output - g_output["output"].output) / delta
    print('Graph Output ',  g_output["output"].output, ' Gradient:', g_output['hidden'].gradients[1][i,j],
          ' Emperical Grad:' , emp_grad)

    graph.train({'input':batch, 'labels':labels} , 0.000001 , 10 , 20)


def predict_quadratic():
    np.random.seed(0)
    input_dimension = 20
    class_count = 1
    batch_size = 4000

    h1 = HiddenLayer(input_dimension + 1, 10, 'relu')
    h2 = HiddenLayer(11, 1, 'linear')

    data = np.random.rand(batch_size, input_dimension)
    out = np.sum(data * data + 5 * data, 1).reshape(batch_size, 1)
    #for i in range(batch_size):
    #    data[i, -1] = 1

    graph = GraphModel()
    graph.add_input('input')
    graph.add_input('target')
    graph.add_node('input_bias', BiasLayer() , 'input' )
    graph.add_node('h1', h1 , 'input_bias')
    graph.add_node('h1_bias', BiasLayer() , 'h1' )
    graph.add_node('h2' , h2 , 'h1_bias')
    graph.add_node('rms' , RMSLayer(), 'target',  'h2')
    graph.add_node('out' , BatchSumLayer(),  'rms')
    graph.compile()
    perf = []
    perf += graph.train({'input':data, 'target':out} , 0.00001, 400 , 10)
    perf+= graph.train({'input':data, 'target':out} ,  0.000001, 400 , 10)

    net_data = graph.run_forward({'input':data, 'target':out})
    print(net_data['out'].output)

    plot_performance(perf)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(file_names , n = 0  ):
    batch_data = []
    batch_label = []
    for f in file_names:
        dict  = unpickle(f)
        batch_data.append(dict[b'data'])
        batch_label.append(dict[b'labels'])

    data = np.concatenate(tuple(batch_data) , 0)
    label_ind = np.concatenate(tuple(batch_label) , 0)
    labels = np.zeros((label_ind.shape[0] , 10))
    ind = 0
    for l in label_ind:
        labels[ind , l] = 1
        ind += 1
    colors = np.split(data, [1024, 2048 ], 1)
    data = np.ndarray.astype(colors[1] + colors[2] + colors[0] , 'float64')
    data -=  np.mean(data, 0)
    if n <= 0:
        n = data.shape[0]
        return  data, labels
    else:
        return data[0:n , :], labels[0:n , :]


def test_cifar10():
    #np.random.seed(0)
    data, labels = load_cifar10(['./data/data_batch_1','./data/data_batch_2','./data/data_batch_3','./data/data_batch_4',
                                 './data/data_batch_5'])
    temp_file = './data/cfar_temp_network.dat'
    if os.path.exists(temp_file):
        with open(temp_file, 'rb') as f:
            print('Loading from the file:' , temp_file)
            graph = pickle.load(f)
            backup_file = temp_file + '.' + str(datetime.datetime.now()).replace(":" , "_").replace('-' , '_').replace('.' , '_')
            print('Make a copy:' , backup_file)
            shutil.copy(temp_file , backup_file)
    else:
        graph = GraphModel()
        graph.add_input("input")
        graph.add_input("labels")
        graph.add_node('input_bias' , BiasLayer(), 'input')
        graph.add_node("hidden_1", HiddenLayer(1024 + 1 , 128, 'relu'), "input_bias")
        graph.add_node("hidden_1_bias" , BiasLayer() , 'hidden_1')
        graph.add_node("hidden_2", HiddenLayer(128 + 1 , 64, 'relu'), "hidden_1_bias")
        graph.add_node("hidden_2_bias" , BiasLayer() , 'hidden_2')
        graph.add_node("hidden_3", HiddenLayer(64 + 1 , 10, 'linear'), "hidden_2_bias")
        graph.add_node("softmax" , SoftmaxLayer() , "hidden_3")
        graph.add_node("cross_ent" , CrossEntropyLayer() , "labels" , "softmax" )
        graph.add_node("output" , BatchSumLayer(), "cross_ent")
        graph.compile()

    perf = []
    test_data, test_labels = load_cifar10(['./data/test_batch'], 5000)
    for i  in range(0 , 200):
        perf +=  graph.train({'input':data, 'labels':labels}, 0.001,  64 , 1)

        net_data = graph.run_forward({'input':test_data, 'labels':test_labels})
        print_precision(test_labels, net_data , 'softmax')
    graph.save_to_file(temp_file)

    # perf += graph.train({'input':data, 'labels':labels}, 0.000005,  128 , 300)
    # graph.save_to_file(temp_file)
    # perf += graph.train({'input':data, 'labels':labels}, 0.00008,  128 , 300)
    # graph.save_to_file(temp_file)
    # perf += graph.train({'input':data, 'labels':labels}, 0.00005,  128 , 300)
    # graph.save_to_file(temp_file)



    #net_data = graph.run_forward({'input':data, 'labels':labels})
    #print_precision(labels, net_data , 'softmax')
    data, labels = load_cifar10(['./data/test_batch'])
    net_data = graph.run_forward({'input':data, 'labels':labels})
    print_precision(labels, net_data , 'softmax')

    plot_performance(perf)


def plot_performance(perf):
    plt.plot(perf)
    plt.ylabel('Cross Entropy')
    plt.xlabel('Batch Run')
    plt.show()


def test_cifar10_rms():
    np.random.seed(0)
    data, labels = load_cifar10(100)
    graph = GraphModel()
    graph.add_input("input")
    graph.add_input("labels")
    graph.add_node('input_bias' , BiasLayer(), 'input')
    graph.add_node("hidden_1", HiddenLayer(1024 + 1 , 128, 'relu'), "input_bias")
    graph.add_node("hidden_1_bias" , BiasLayer() , 'hidden_1')
    graph.add_node("hidden_2", HiddenLayer(128 + 1 , 10, 'relu'), "hidden_1_bias")
    graph.add_node("rms" , RMSLayer() , "labels",  "hidden_2")
    graph.add_node("output" , BatchSumLayer(), "rms")
    graph.compile()
    graph.train({'input':data, 'labels':labels}, 0.000001,  50 , 10)
    graph.train({'input':data, 'labels':labels}, 0.0000001,  100 , 100)
    graph.train({'input':data, 'labels':labels}, 0.00000001,  500 , 100)

    net_data = graph.run_forward({'input':data, 'labels':labels})

    print_precision(labels, net_data, 'hidden_2')


def print_precision(labels, net_data , out_layer ):
    predict = np.argmax(net_data[out_layer].output, 1)
    correct = np.sum(np.where(predict == np.argmax(labels, 1), 1, 0))
    print('Correct: ', correct, '/' , labels.shape[0])


if __name__ == "__main__":
    #calculate_emperical_gradient_simple()
    #calculate_emperical_gradient()
    #predict_quadratic()
    test_cifar10()

