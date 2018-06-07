import random
import math
random.seed(1)
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
        self.layers[0].reset()
        self.layers[-1].reset()
    
    @staticmethod
    def nonlin(x,derivative=False):
        if derivative == True: return x * (1 - x) #Assumption x in nonlin form
        else: return 1 / (1 + e(x))
    
    def feedForward(self,inputs):
        assert(len(inputs) == self.inputCount)
        activationLayers = []
        currentActivations = inputs
        for layerNum,layer in enumerate(self.layers[0:-1]):
            newActivations = layer.getActivationLayer(currentActivations)
            activationLayers.append(newActivations)
            currentActivations = newActivations
        return activationLayers

    def train(self,inputs,outputs):
        actualOutputs = outputs
        activationLayers = self.feedForward(inputs)
        guessedOutputs = activationLayers[-1]
        # print(guessedOutputs)

        #Current Weight Training -> layer[1][0]
        target = actualOutputs[0]
        activation = guessedOutputs[0]
        previousActivation = activationLayers[1][0]

        # print(target,activation,previousActivation)

        errorDerivative = NeuralNetwork.errorDerivative(target,activation)
        nonlinDerivative = NeuralNetwork.nonlin(activation,derivative=True)
        previousActivationDerivative = previousActivation
        weightDerivative = errorDerivative * nonlinDerivative * previousActivationDerivative
        self.layers[1].neurons[0].weight -= weightDerivative * 0.5
        

        # previousLayer = activationLayers[-2]
        # for layerNum,currentLayer in enumerate(activationLayers[-1:0:-1]):
        #     for index,activation in enumerate(currentLayer):
        #         previousActivation = previousLayer[0] #current only checking for first weight
        #         target = guessedOutputs[-1]
        #         #Derivatives
        #         errorDerivative = NeuralNetwork.errorDerivative(target,activation)
        #         nonlinDerivative = NeuralNetwork.nonlin(activation,derivative=True)
        #         previousActivationDerivative = previousActivation

        #         weightDerivative = errorDerivative * nonlinDerivative * previousActivationDerivative
            


    
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
        assert(len(activations) == self.layerSize)
        for index in range(self.layerSize):
            currentNeuron = self.neurons[index]
            previousLayerActivation = activations[index]
            currentActivation = currentNeuron.getActivation(previousLayerActivation)
            newActivations.append(currentActivation)
        return newActivations
    
    def reset(self):
        '''
        Weight -> 1
        Bias -> 0
        '''
        for neuron in self.neurons:
            neuron.weight = 1
            neuron.bias = 0

if __name__ == '__main__':
    print("In Main")
    NN = NeuralNetwork([1,1,1])
    # NN.feedForward([1])
    for i in range(10000):
        NN.train([1],[0.1])
    print(NN.feedForward([1]))
else:
    print("Not in Main")
