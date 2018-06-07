import random
from math import *
import numpy as np

random.seed(1)

class Neuron:
    def __init__(self,weight,bias,layerNum,layerIndex,inputLayer=False,outputLayer=False):
        #Relationships
        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.parents = []
        self.children = []
        #Weights and Biases
        self.weight = weight
        self.bias = bias
        #Information
        self.layerNum = layerNum
        self.layerIndex = layerIndex
        self.name = '(' + str(self.layerNum) + "," + str(self.layerIndex) + ')'
        #Change Weights for input Neurons
        if self.inputLayer:
            self.weight = 1
            self.bias = 0

    def setRelationships(self,layers):
        """
        Sets the parent and child neurons
        """
        if self.inputLayer == False:
            self.parents = layers[self.layerNum-1]
        if self.outputLayer == False:
            self.children = layers[self.layerNum+1]

    def __str__(self):
        string = ''
        string += self.name + '\n'
        string += "Parents: " + ', '.join([(lambda x:x.name)(x) for x in self.parents]) + '\n'
        string += "Children: " + str(len(self.children)) + '\n'
        string += "Weight: " + str(self.weight) + '\n'
        string += "Bias: " + str(self.bias) + '\n'
        return string

class NeuralNetwork:
    def __init__(self,networkSize,learningRate,data):
        #NN information
        self.layers = self.createLayers(networkSize)
        self.layerCount = len(self.layers)
        self.learningRate = learningRate
        #Testing Data
        self.trainingDataCount = int(ceil(len(data) * 0.8))
        self.trainingData = data[:self.trainingDataCount:]
        self.testingData = data[self.trainingDataCount:]

    def __str__(self):
        """
        Representation of NN
        """
        maxLength = max([(lambda layer: len(layer))(layer) for layer in self.layers])
        string = ""
        for layerNum, layer in enumerate(self.layers):
            offset = int(maxLength - len(layer))
            print(' ' * offset + '-'.join(['*'] * len(layer)) + '\n')
        return string

    def createLayers(self,networkSize):
        """
        Creates all the layers and neurons in NN
        """
        layers = []
        for layerNum,layerSize in enumerate(networkSize):
            layer = []
            for layerIndex in range(layerSize):
                layer.append(Neuron(random.uniform(-1,1),random.uniform(-1,1),layerNum,layerIndex,
                            inputLayer=layerNum==0,outputLayer=layerNum==len(networkSize)-1))
            layers.append(layer)
        for layer in layers:
            for neuron in layer:
                neuron.setRelationships(layers)
        return layers

    def forwardPropogation(self,inputs):
        """
        Passes inputs through neural network
        Applies biases and weights and finally gets output layer
        """
        activations = []

        for i,layer in enumerate(self.layers):
            outputs = []
            for neuron in layer:
                totalActivation = 0
                # print(inputs)
                for inputVal in inputs:
                    totalActivation += neuron.weight * inputVal + neuron.bias
                outputs.append(self.sigmoid(totalActivation))
            activations.append(outputs)
            inputs = outputs
        return activations

    def train(self):
        for inputVals,target in self.trainingData[0:1]:
            activations = self.forwardPropogation(inputVals)
            output = activations[-1]
            totalError = 1/2 * (sum(self.getArrayDifference(target,output))**2)
            # print(totalError)
            for outputIndex,outputNeuron in enumerate(self.layers[self.layerCount-1]):
                for layerIndex,neuron in enumerate(self.layers[self.layerCount-2]):
                    derivative1 = -(target[outputIndex] - output[outputIndex])

                    derivative2 = self.sigmoid(target[outputIndex],derivative=True)

                    derivative3 = neuron.weight #need to get neuron value

                    finalDerivative =  derivative1 * derivative2 * derivative3
                    neuron.weight -= finalDerivative * self.learningRate
        # print(output)




    def sigmoid(self,x,derivative=False):
        """
        Function to set values in array between 0,1 and makes computation easier
        """
        if(derivative==True):
            return x*(1-x)
        else:
            return 1/(1+np.exp(-x))

    def getArrayDifference(self,first,second):
        new = []
        assert(len(first) == len(second))
        for i in range(len(first)):
            new.append(first[i] - second[i])
        return new



if __name__ == '__main__':
    print("Testing NN in main!")
    # If number is even return 1, else return 0
    data = [([i],[i%2]) for i in range(100)] #Data being used
    learningRate = 0.5 #Rate at which NN learns
    networkSize = [1,2,1] #First and last determine size of input and output
    # Current has 1 input, 2 neurons in hidden layer and 1 output
    NN = NeuralNetwork(networkSize,learningRate,data)
    NN.train()
    # print(NN.forwardPropogation([1]))
    # print(NN.forwardPropogation([0]))
