import random
import math
#from matplotlib import pyplot as plt
#import numpy as np

class Network:

    #initialise network with number of inputs, number of hidden nodes and weights and bias' can be assigned if desired
    def __init__(self, no_inputs, no_hidden, learn_rate, weightsH = None, weightsO = None, biasH = None, biasO = None):
        self.noOfInputs = no_inputs
        self.learnRate = learn_rate 

        self.layerHidden = HiddenLayer(no_hidden, biasH) #set hidden layer object with inputted number of hidden nodes and bias if assigned
        self.nodeOutput = OutputNode(biasO) #set output node object with inputted bias if assigned
        
        self.setWeightsH(weightsH) #call function to set weights between inputs and hidden layer, set to assigned weights if inputted
        self.setWeightsO(weightsO) #call function to set weights between hidden layer and output node, set to assigned weights if inputted

    def setWeightsH(self, weightsH): #loop through each hidden node and append a weight to node's weight list for each input
        weight = 0
        for i in range(len(self.layerHidden.nodes)): 
            for j in range(self.noOfInputs):
                if not weightsH:
                    self.layerHidden.nodes[i].weights.append(random.random())
                    self.layerHidden.nodes[i].weights_change.append(0)
                else:
                    self.layerHidden.nodes[i].weights.append(weightsH[weight])
                    self.layerHidden.nodes[i].weights_change.append(0)
                weight += 1
    
    def setWeightsO(self, weightsO): #loop through each hidden node and append weight to output node weight list
        weight = 0
        for i in range(len(self.layerHidden.nodes)):
            if not weightsO:
                self.nodeOutput.weights.append(random.random())
                self.nodeOutput.weights_change.append(0)
            else:
                self.nodeOutput.weights.append(weightsO[weight])
                self.nodeOutput.weights_change.append(0)
            weight += 1

    def forwardPass(self, inputs): #use activation functions to make forward pass through network
        return self.nodeOutput.activation(self.layerHidden.forwardPass(inputs))

    def train(self, inputs, output, epoch):
        self.forwardPass(inputs)

        #Delta for output node with weight decay
        n = 0
        sum_weightsSqr = 0
        for i in range(len(self.layerHidden.nodes)): #loop through all hidden nodes and add their bias and weights squared
            sum_weightsSqr += self.layerHidden.nodes[i].bias*self.layerHidden.nodes[i].bias
            n += 1 #incrementing for every hidden node bias added
            for j in range(len(self.layerHidden.nodes[i].weights)):
                n += 1 #incrementing for every hidden weight added
                sum_weightsSqr += self.layerHidden.nodes[i].weights[j]*self.layerHidden.nodes[i].weights[j]

        for i in range(len(self.nodeOutput.weights)): #loop through output weights and add their square
            n += 1 #incrementing for every output weight added
            sum_weightsSqr += self.nodeOutput.weights[i]*self.nodeOutput.weights[i]
        sum_weightsSqr += self.nodeOutput.bias*self.nodeOutput.bias
        n += 1 #incrementing for the output bias added

        outputDelta = self.nodeOutput.delta(output, epoch, sum_weightsSqr, n, self.learnRate)

        #Delta for hidden nodes
        hiddenDeltas = [0]*len(self.layerHidden.nodes)
        for i in range(len(self.layerHidden.nodes)):
            hiddenDeltas[i] = self.layerHidden.nodes[i].derivative(output)*outputDelta*self.nodeOutput.weights[i]
        
        #Update weights and bias for hidden to output node using momentum calculation
        self.nodeOutput.old_weights = self.nodeOutput.weights #Update old weights to be used for change calculation
        for i in range(len(self.nodeOutput.weights)):
            self.nodeOutput.weights[i] += self.learnRate*outputDelta*self.nodeOutput.output + self.nodeOutput.weights_change[i]*0.9
        
        self.nodeOutput.old_bias = self.nodeOutput.bias #Update old bias to be used for change calculation
        self.nodeOutput.bias += self.learnRate*outputDelta*self.nodeOutput.output + self.nodeOutput.bias_change*0.9

        #Save change in weights and bias for hidden to output node
        for i in range(len(self.nodeOutput.weights)):
            self.nodeOutput.weights_change[i] = self.nodeOutput.weights[i]-self.nodeOutput.old_weights[i]

        self.nodeOutput.bias_change = self.nodeOutput.bias-self.nodeOutput.old_bias

        #Update weights and bias for input to hidden using momentum calculation
        for i in range(len(self.layerHidden.nodes)):
            self.layerHidden.nodes[i].old_weights = self.layerHidden.nodes[i].weights #Update old weights to be used for change calculation
            self.layerHidden.nodes[i].old_bias = self.layerHidden.nodes[i].bias #Update old bias to be used for change calculation
            for j in range(len(self.layerHidden.nodes[i].weights)):
                self.layerHidden.nodes[i].weights[j] += self.learnRate*hiddenDeltas[i]*self.layerHidden.nodes[i].output + self.layerHidden.nodes[i].weights_change[j]*0.9
            self.layerHidden.nodes[i].bias += self.learnRate*hiddenDeltas[i]*self.layerHidden.nodes[i].output + self.layerHidden.nodes[i].bias_change*0.9

        #Save change in weights and bias for input to hidden 
        for i in range(len(self.layerHidden.nodes)):
            for j in range(len(self.layerHidden.nodes[i].weights)):
                self.layerHidden.nodes[i].weights_change[j] = self.layerHidden.nodes[i].weights[j]-self.layerHidden.nodes[i].old_weights[j]
            self.layerHidden.nodes[i].bias_change = self.layerHidden.nodes[i].bias-self.layerHidden.nodes[i].old_bias

    def raw_error(self, inputs, correct_output): #calculate raw error from given output
        self.forwardPass(inputs)
        return correct_output-self.nodeOutput.output
    
    def MSE(self, inputs, correct_output): #calculate error of output node using mean square error method
        self.forwardPass(inputs)
        return 0.5*(correct_output-self.nodeOutput.output)**2

    def annealing(self, p, q, r, x): #calculate new learn rate using parameters inputted using annealing method
        self.learnRate = p+(q-p)(1-(1/(1+math.exp(10-(20*x/r)))))
    
class HiddenLayer:
    def __init__(self, no_nodes, bias):
        self.bias = bias if bias else random.random() 
        self.nodes = []
        for i in range(no_nodes): #instantiate each node object with same bias
            self.nodes.append(Node(self.bias))
    
    def output(self): #save each nodes output to a list
        o = []
        for n in self.nodes:
            o.append(n.output)
        return o
    
    def forwardPass(self, inputs): #save each activation function output to a list
        o = []
        for n in self.nodes:
            o.append(n.activation(inputs))
        return o

class Node:
    def __init__(self, bias): #initialise node with bias and weights
        self.bias = bias #every node in a layer has the same bias
        self.old_bias = 0
        self.bias_change = 0
        self.weights = []
        self.old_weights = []
        self.weights_change = []
    
    def activation(self, inputs):
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)): #total up node weighted inputs
            total += self.inputs[i]*self.weights[i]
        self.output = 1/(1+math.exp(-(total+self.bias))) #use sigmoid function to calculate output
        return self.output

    def derivative(self, correctOutput): #calculate 1st derivative of sigmoid function for this node
        return self.output*(1-self.output)

class OutputNode:
    def __init__(self, bias):
        self.bias = bias if bias else random.random() #initialise output node with bias
        self.old_bias = 0
        self.bias_change = 0
        self.weights = []
        self.old_weights = []
        self.weights_change = []

    def activation(self, inputs):
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)): #total weighted inputs on output node
            total += self.inputs[i]*self.weights[i]
        self.output = 1/(1+math.exp(-(total+self.bias))) #use sigmoid function on output node 
        return self.output
    
    def delta(self, correctOutput, epoch, sum_weightsSqr, n, p): #calculate output delta with weight decay calculation
        upsilon = 1/(p*epoch)
        omega = (n*0.5)*sum_weightsSqr
        return ((correctOutput-self.output)+(upsilon*omega))*(self.output*(1-self.output))
    

    
n = Network(2, 2, 0.1)
x = []
y = []

result = 0
i = 0
for i in range(2094):
#while result != 1:
    n.annealing(0.01, 0.1, 2094, i)
    n.train([0.9, 0.1], 1, i)
    print("Epochs " + str(i), n.MSE([1, 0], 1))
    y.append(i)
    x.append(n.raw_error([1, 0], 1))
    result = n.nodeOutput.output
    i += 1
#xp = np.array(x)
#yp = np.array(y)

#plt.plot(yp, xp) 
#plt.show()