from layers import *
import pickle

class LayerData:

    def __init__(self, inputs, output, loss_gradient, gradients):
        self.inputs = inputs
        self.output = output
        self.loss_gradient = loss_gradient
        self.gradients = gradients


class NetworkNode:

    def __init__(self, layer_identifier, layer, parents):
        self.layer = layer
        self.parents = parents
        self.identifier = layer_identifier


def slice_inputs(sub_indices, inputs):
    ret = {}
    for k, v in inputs.items():
        ret[k] = v[sub_indices, :]
    return ret


def build_batch(n, batch_size):
    perm = np.random.permutation(n)
    ret = []
    ind = 0

    while ind < n:
        next = ind + batch_size
        if next > n:
            next = n
        ret.append(perm[ind:next])
        ind = next

    return ret


def print_gradiant_info(loss_gradient):
    print('Max:', np.max(loss_gradient) , ' Min:' , np.min(loss_gradient), ' Mean:' , np.mean(loss_gradient))


class GraphModel:

    def __init__(self):
        self.nodes = {}
        self.inputs = set()

    def save_to_file(self, filename):
        pickle.dump(self , open(filename, 'wb'))

    def train(self, network_inputs, learning_rate, batch_size=200, epoch_count=20):
        data_count = -1
        perf = []
        for k, v in network_inputs.items():
            if data_count == -1:
                data_count = v.shape[0]
            if data_count != v.shape[0]:
                raise Exception("All input first dimension must agree")
        for epoch in range(1, epoch_count + 1):
            slices = build_batch(data_count, batch_size)
            batch = 0
            print('Epoch ', epoch , end='')
            x = int(len(slices) / 20)
            if x == 0:
                x = 1

            for sl in slices:
                batch += 1
                if (batch % x == 0):
                    print('=' , end='')
                net_data = self.run_forward(slice_inputs(sl, network_inputs))
                perf.append(net_data[self.node_order[-1].identifier].output)
                self.run_backward(net_data)
                for nid in self.nodes:
                    if nid in self.inputs:
                        continue
                    self.nodes[nid].layer.apply_backprop(net_data[nid].gradients[-1], learning_rate)
            print('Epoch Performance:', np.mean(perf[-len(slices):]))
        return perf
    def add_node(self, layer_identifier, layer, *parent_layers):
        if layer_identifier in self.nodes:
            raise Exception("Layer already created:", layer_identifier)
        parents = []
        for parent_layer in parent_layers:
            if parent_layer not in self.nodes:
                raise Exception("Cannot find the parent layer ", parent_layer)
            parents.append(self.nodes[parent_layer])
        self.nodes[layer_identifier] = NetworkNode(layer_identifier, layer, parents)

    def add_input(self, input_name):
        self.nodes[input_name] = NetworkNode(input_name, None, None)
        self.inputs.add(input_name)

    def compile(self):
        """This method should be called after network topology is established.
        Basically it calculates the topology of the network for further use"""
        self.node_order = []
        processed_nodes = set()
        for l in self.inputs:
            self.node_order.append(self.nodes[l])
            processed_nodes.add(l)

        something_found = True
        while something_found and len(self.node_order) < len(self.nodes):
            something_found = False
            for id, node in self.nodes.items():
                if id not in processed_nodes:
                    all_ok = True
                    inputs = []
                    for p in node.parents:
                        if p.identifier not in processed_nodes:
                            all_ok = False
                            break
                    if all_ok:
                        processed_nodes.add(id)
                        self.node_order.append(node)

    def run_forward(self, input_values):
        if len(self.inputs.symmetric_difference(input_values.keys())) != 0:
            raise Exception("Please provide right inputs for this network:Missing or additional inputs detected",
                            self.inputs.symmetric_difference(input_values.keys()))
        layer_data = {}
        for k, v in input_values.items():
            layer_data[k] = LayerData([v], v, None, None)
        for n in self.node_order:
            if n.identifier in layer_data:
                continue

            inputs = []
            for p in n.parents:
                inputs.append(layer_data[p.identifier].output)
            layer_data[n.identifier] = LayerData(inputs, n.layer.forward_pass(inputs), None, None)
        return layer_data

    def run_backward(self, layers_data):

        processed = []
        for n in reversed(self.node_order):
            if n.identifier in self.inputs:
                continue
            if len(processed) == 0:
                loss_gradient = 1
            else:
                loss_gradient = np.zeros(layers_data[n.identifier].output.shape)
                for p in processed:
                    ind = 0
                    for parent in p.parents:
                        if parent.identifier == n.identifier:
                            loss_gradient += layers_data[p.identifier].gradients[ind]
                        ind += 1
            layers_data[n.identifier].loss_gradient = loss_gradient
            #print('Layer:', n.identifier)
            #print_gradiant_info(loss_gradient)
            layers_data[n.identifier].gradients = n.layer.backward_pass(loss_gradient, layers_data[n.identifier].inputs)
            processed.append(n)
        return layers_data



class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, new_layer):
        self.layers.append(new_layer)

    def run_batch(self, batch):
        layer_outputs = []
        layer_outputs.append(batch)
        next_input = batch
        for l in self.layers:
            out = l.forward_pass([next_input])
            layer_outputs.append(out)
            next_input = out
        return layer_outputs

    def backward_run(self, layer_outputs):
        backward_gradients = []
        backward_gradients.insert(0, (1, []))  # The first input gradient
        for i in range(len(self.layers)):
            next_grad = self.layers[len(self.layers) - i - 1].backward_pass(backward_gradients[0][0],
                                                                            [layer_outputs[len(self.layers) - i - 1]])
            backward_gradients.insert(0, next_grad)
        return backward_gradients
