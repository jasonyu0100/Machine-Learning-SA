#Libraries
import random
import math
#Random Seed for consistency
random.seed(1)
#Renamed Functions
randomValue = random.uniform
e = math.exp

class NeuralNetwork:
    """
    Holds network layers, preferences and controls training
    """
    def __init__(self,layerInformation):
        self.inputCount = layerInformation[0]
        self.outputCount = layerInformation[-1]
        self.layers = [NetworkLayer(layerNum,layerSize) for layerNum,layerSize in enumerate(layerInformation)]
        self.layers[0].reset() #Input Layer Has no Affect
    
    @staticmethod
    def nonlin(x,derivative=False):
        if derivative == True: return x * (1 - x) #Assumption x in nonlin form
        else: return 1 / (1 + e(x))
    
    def feedForward(self,inputs):
        assert(len(inputs) == self.inputCount)
        activationLayers = [inputs] #Initial Activations are inputs
        for layerNum,layer in enumerate(self.layers[1:]):
            newActivations = layer.getActivationLayer(activationLayers[-1])
            activationLayers.append(newActivations)
        return activationLayers

    def train(self,inputs,outputs):
        actualOutputs = outputs
        activationLayers = self.feedForward(inputs)
        guessedOutputs = activationLayers[-1]
        for index,activation in enumerate(guessedOutputs):
            target = actualOutputs[index]
            activation = guessedOutputs[index]
            for previousActivation in activationLayers[-2]: #Previous Activations from second last layer
                #Chain Function Derivatives
                errorDerivative = NeuralNetwork.errorDerivative(target,activation)
                nonlinDerivative = NeuralNetwork.nonlin(activation,derivative=True)
                previousActivationDerivative = previousActivation
                #Final Derivative
                weightDerivative = errorDerivative * nonlinDerivative * previousActivationDerivative
                self.layers[-1].neurons[0].weight -= weightDerivative * 0.5

        

    @staticmethod
    def calculateTotalError(actualOutputs,guessedOutputs):
        return sum(NeuralNetwork.calculateError(target,guess) for target,guess in zip(actualOutputs,guessedOutputs))

    @staticmethod
    def calculateError(actual,guess):
        return (0.5 * (target-guess)**2)

    @staticmethod
    def errorDerivative(target,guess):
        return target - guess

class Neuron:
    """
    Holds weights and biases
    """
    def __init__(self,layerNum,index):
        self.layerNum = layerNum
        self.index = index
        self.weight = randomValue(-1,1)
        self.bias = 0#randomValue(-1,1)

    def getActivation(self,activation):
        activation = self.weight * activation + self.bias
        nonlinActivation = NeuralNetwork.nonlin(activation)
        return nonlinActivation
        
class NetworkLayer:
    """
    Contains neurons 
    """
    def __init__(self,layerNum,layerSize):
        self.layerNum = layerNum
        self.layerSize = layerSize
        self.neurons = [Neuron(layerNum,index) for index in range(layerSize)]

    def getActivationLayer(self,activations):
        newActivations = []
        for currentNeuron in self.neurons:
            netActivation = sum(currentNeuron.getActivation(previousLayerActivation) for previousLayerActivation in activations)
            averagedActivation = netActivation / len(activations)
            newActivations.append(averagedActivation)
        return newActivations
    
    def reset(self):
        '''
        Weight -> 1
        Bias -> 0
        '''
        for neuron in self.neurons:
            neuron.weight = 1
            neuron.bias = 0

#In this current version only the weights of the output layer are being changed

if __name__ == '__main__':
    print("In Main")
    NN = NeuralNetwork([1,1,1]) #Two Layer Neural Network
    print(NN.feedForward([1]))
    for i in range(1000):
        NN.train([1],[0])
    print(NN.feedForward([1]))
else:
    print("Not in Main")
