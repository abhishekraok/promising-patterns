nc_labels = []  # a list of string, where each string is a label of new class node
nc_list = []  # a list of NewClass nodes
top_N  = 5  # number of features to select for each new class

class KadNet():
   """The main class containing all the KadNodes.""" 
   def __init__(self, labels_list=['Default1','Default2','Default3']):
       self.labels_list = labels_list
       self.nodes = [Kadnode(label=i) for i in self.labels_list]

    def check_outputs(self):
        """ Displays pairwise node output values and their corresponding output
        values"""
        print 'Displaying all the activation values'
        for i in self.nodes:
            print 'Label = ', i.label, ': Output = ', i.output


class Kadnode():
    """ A new class, a new label created from function of old"""
    def __init__(self, label=None, output=None):
        self.label = label if not label is None else None
        self.output= output if not output is None else None
        self.inputs = []
        self.weights = []
        
    def get_activation(self):
        if self.output is not None:
            return self.output
        else:
            output = sum([get_activation_by_label(self.inputs[i]) *
                self.weights[i] for i in range(len(self.inputs))])

def activate_all(in_net, filename, new_classes):
    """ Activate all the nodes for given file with name
    filename."""
    feat7, feat8 = extract_caffe_features(in_net, filename)
    feat8_nc = [NewClass(caffe_labels[i],feat8)] 
    nc_activations = [nc_node.get_activation() for nc_node in new_classes]
    all_activated_nodes =  feat8_nc + nc_activations
    return all_activated_nodes

def get_all_activation_numbers(all_activated_nodes):
    """ Get the activation values of all nodes"""
    all_activations_numbers = [i.output for i in all_activated_nodes]
    return all_activations_numbers

def create_new(activations_list, new_label):
    """ pick the highest activated features and create new class"""
    top_activations_index = np.array(activations_list).argsort()[-1:-top_N:-1]
    new_node = NewClass(new_label)
    
def get_activation_by_label(node_label):
    """ Get the activation function of the current node"""
    
    if node_label in caffe_labels:
        caffe_index = caffe_labels.index(node_label)
        return feat8[caffe_index]
    else:
        nc_node = nc_list[nc_labels.index(node_label)]
        return nc_node.get_activation()
    
if __name__ == '__main__':
    main_list = KadNet()
    main_list.check



