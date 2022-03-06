import random
import math

class Network:
    learnRate = 0.1

    #initialise network with number of inputs, number of hidden nodes and weights and bias' can be assigned if desired
    def __init__(self, no_inputs, no_hidden, weightsH = None, weightsO = None, biasH = None, biasO = None):
        self.noOfInputs = no_inputs

        self.layerHidden = HiddenLayer(no_hidden, biasH) #set hidden layer object with inputted number of hidden nodes and bias if assigned
        self.nodeOutput = OutputNode(biasO) #set output node object with inputted bias if assigned
        
        self.setWeightsH(weightsH) #call function to set weights between inputs and hidden layer, set to assigned weights if inputted
        self.setWeightsO(weightsO) #call function to set weights between hidden layer and output node, set to assigned weights if inputted

    def setWeightsH(self, weightsH): #loop through each hidden node and append a weight to node's weight list for each input
        weight = 0
        for i in range(len(self.layerHidden)): 
            for j in range(self.noOfInputs):
                if not weightsH:
                    self.layerHidden.nodes[i].weights.append(random.random())
                else:
                    self.layerHidden.nodes[i].weights.append(weightsH[weight])
                weight += 1
    
    def setWeightsO(self, weightsO): #loop through each hidden node and append weight to output node weight list
        weight = 0
        for i in range(len(self.layerHidden)):
            if not weightsO:
                self.nodeOutput.weights.append(random.random())
            else:
                self.nodeOutput.weights.append(weightsO[weight])
            weight += 1

    def forwardPass(self, inputs): #use activation functions to make forward pass through network
        return self.nodeOutput.activation(self.layerHidden.forwardPass(inputs))

    def train(self, inputs, output):
        self.forwardPass(inputs)

        #Delta for output node
        outputDelta = self.nodeOutput.delta(output)

        #Delta for hidden nodes
        hiddenDeltas = [0]*len(self.layerHidden.nodes)
        for i in range(len(self.layerHidden.nodes)):
            hiddenDeltas[i] = self.layerHidden.nodes[i].derivative(output)*outputDelta*self.nodeOutput.weights[i]
        
        #Update weights and bias for hidden to output node
        for i in range(len(self.nodeOutput.weights)):
            self.nodeOutput.weights[i] += self.learnRate*outputDelta*self.nodeOuput.output
        
        self.nodeOutput.bias += self.learnRate*outputDelta*self.nodeOutput.output

        #Update weights and bias for input to hidden
        for i in range(len(self.layerHidden.nodes)):
            for j in range(len(self.layerHidden.nodes[i].weights)):
                self.layerHidden.nodes[i].weights[j] += self.learnRate*hiddenDeltas[i]*self.layerHidden.nodes[i].output
            self.layerHidden.nodes[i].bias += self.learnRate*hiddenDeltas[i]*self.layerHidden.nodes[i].ouput

    def error(self, correct_output):
        return correct_output-self.nodeOutput.output

    

class HiddenLayer:
    def __init__(self, no_nodes, bias):
        self.bias = bias if bias else random.random() 
        self.nodes = []
        for i in range(no_nodes): #instantiate each node object with same bias
            self.nodes.append(Node(self.bias))
    
    def output(self):
        o = []
        for n in self.nodes:
            o.append(n.output)
        return o
    
    def passForward(self, inputs):
        o = []
        for n in self.nodes:
            o.append(n.activation(inputs))
        return o

class Node:
    def __init__(self, bias): #initialise node with bias and weights
        self.bias = bias #every node in a layer has the same bias
        self.weights = []
    
    def activation(self, inputs):
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)): #total up node weighted inputs
            total += self.inputs[i]*self.weights[i]
        self.output = 1/(1+math.exp(-(total+self.bias))) #use sigmoid function to calculate output
        return self.output
    
    def error(self, correctOutput): #calculate node error 
        return correctOutput-self.output

    def derivative(self, correctOutput):
        return self.output*(1-self.output)

class OutputNode:
    def __init__(self, bias):
        self.bias = bias if bias else random.random() #initialise output node with bias
        self.weights = []

    def activation(self, inputs):
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)): #total weighted inputs on output node
            total += self.inputs[i]*self.weights[i]
        self.output = 1/(1+math.exp(-(total+self.bias))) #use sigmoid function on output node 
        return self.output
    
    def delta(self, correctOutput):
        return (correctOutput-self.output)*(self.output*(1-self.output))
    
nn = Network(2, 2, 1)

for i in range(2094):
    nn.train([1, 0], [1])
    print("Epochs " + str(i), nn.error([1]))