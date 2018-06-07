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
        self.layerInformation = layerInformation
        self.inputCount = layerInformation[0]
        self.outputCount = layerInformation[-1]
        self.layers = [NetworkLayer(layerNum,layerSize,self) for layerNum,layerSize in enumerate(layerInformation)]
        self.layers[0].reset() #Input Layer Has no Affect
    
    @staticmethod
    def nonlin(x,derivative=False):
        if derivative == True: return x * (1 - x) #Assumption x in nonlin form
        else: return 1 / (1 + e(x))
    
    def feedForward(self,inputs):
        assert(len(inputs) == self.inputCount)
        activationLayers = [inputs] #Initial Activations are inputs
        for layer in self.layers[1:]:
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
                self.layers[-1].neurons[index].weight -= weightDerivative * 0.5
    
    # def neuronErrorContributionDerivative(self,neuron):
    #     errorDerivative = NeuralNetwork.errorDerivative(target,activation)
    #     nonlinDerivative = NeuralNetwork.nonlin(activation,derivative=True)
    #     weightDerivative = neuron.weight

    @staticmethod
    def calculateTotalError(actualOutputs,guessedOutputs):
        return sum(NeuralNetwork.calculateError(target,guess) for target,guess in zip(actualOutputs,guessedOutputs))

    @staticmethod
    def calculateError(actual,guess):
        return (0.5 * (actual-guess)**2)

    @staticmethod
    def errorDerivative(target,guess):
        return target - guess

    def getError(self,inputs,actual):
        guessed = self.feedForward(inputs)[-1]
        totalError = self.calculateTotalError(actual,guessed)
        return totalError

    def display(self):
        for layer in self.layers:
            print("Layer Num: " + str(layer.layerNum))
            layer.display()

class Neuron:
    """
    Holds weights and biases
    """
    def __init__(self,layerNum,index,previousLayerSize):
        #Should Have multiple weights
        self.layerNum = layerNum
        self.index = index
        self.weight = randomValue(-1,1)
        self.bias = 0#randomValue(-1,1)

        self.previousLayerSize = previousLayerSize

    def weightsAndBiases(self,previousLayerSize):
        #If previous layer size
        if previousLayerSize == None:
            #Must be input variable
            pass
        else:
            self.weights = [randomValue(-1,1) for connection in range(previousLayerSize)]
            self.biases = [randomValue(-1,1) for connection in range(previousLayerSize)]

    def getActivation(self,activation,neuronIndex):
        #Using Neuron Index it should use corresponding weight and bias
        activation = self.weight * activation + self.bias
        nonlinActivation = NeuralNetwork.nonlin(activation)
        return nonlinActivation

    def display(self):
        print("W: {} B: {}".format(self.weight,self.bias))
        
class NetworkLayer:
    """
    Contains neurons 
    """
    def __init__(self,layerNum,layerSize,NN):
        self.NN = NN
        self.layerNum = layerNum
        self.layerSize = layerSize
        self.neurons = [Neuron(layerNum,index,self.getPreviousLayerSize()) for index in range(layerSize)]

    def getActivationLayer(self,activations):
        newActivations = []
        for currentNeuron in self.neurons:
            netActivation = sum(currentNeuron.getActivation(previousLayerActivation,neuronIndex) for neuronIndex,previousLayerActivation in enumerate(activations))
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
    
    def display(self):
        for neuron in self.neurons:
            neuron.display()

    def getPreviousLayerSize(self):
        if self.layerNum > 0: return self.NN.layerInformation[self.layerNum - 1]
        else: return None


#In this current version only the weights of the output layer are being changed

if __name__ == '__main__':
    print("In Main")
    iterations = 1000
    inputs = [1]
    outputs = [0,0]
    NN = NeuralNetwork([1,2]) #Two Layer Neural Network
    print(NN.feedForward(inputs))
    for i in range(iterations):
        NN.train(inputs,outputs)
    print(NN.feedForward(inputs))
else:
    print("Not in Main")
