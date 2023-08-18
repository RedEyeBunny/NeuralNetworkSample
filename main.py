import numpy as np
import scipy.special
import matplotlib.pyplot as plt
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,output,lr):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = output
        self.lr = lr
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes,0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train (self,inputs_list,targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        #output layer error = (target-actual)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)
        #update the weights for the links btw the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    def query(self,input_list):
        #convert list into 2d array
        input = np.array(input_list,ndmin = 2).T
        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih,input)
        #calculate signals out of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calc signals in final output layer
        final_input = np.dot(self.who,hidden_outputs)
        #calc signals from the final output layer
        final_outputs = self.activation_function(final_input)
        print (final_outputs)

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file = open("/home/chilltoast/Downloads/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array,cmap="Greys", interpolation='None')
print (n.query((np.asfarray(all_values[1:])/255.0 * 0.99)+0.01))
plt.show()
