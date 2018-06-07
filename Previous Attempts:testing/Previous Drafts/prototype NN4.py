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
        self.learningRate = 0.5
        self.inputCount = layerInformation[0]
        self.outputCount = layerInformation[-1]
        self.layers = [NetworkLayer(layerNum,layerSize,self) for layerNum,layerSize in enumerate(layerInformation)]
        self.delta = {}
    
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

    def train(self,inputs,actualOutputs):
        """
        Trains Neural Network with test data
        """
        activationLayers = self.feedForward(inputs)
        guessedOutputs = activationLayers[-1]
        errorDerivatives = [NeuralNetwork.errorDerivative(target,guess) for target,guess in zip(actualOutputs,guessedOutputs)]
        nonlinDerivatives = [NeuralNetwork.nonlin(guess,derivative=True) for target,guess in zip(actualOutputs,guessedOutputs)]
        errorSignals = [(errorDerivative * nonlinDerivative) for errorDerivative,nonlinDerivative in zip(errorDerivatives,nonlinDerivatives)]
        #Output Neuron BackPropogation
        curretLayerNum = len(self.layers) - 1
        currentLayer = activationLayers[curretLayerNum]
        for neuronIndex in range(len(currentLayer)): #Current Neuron being trained
            errorSignal = errorSignals[neuronIndex]
            for weightIndex in range(len(activationLayers[curretLayerNum-1])): #Previous Activations from second last layer
                previousActivationDerivative = activationLayers[curretLayerNum-1][weightIndex]
                self.backPropogation(neuronIndex,weightIndex,curretLayerNum,errorSignal,previousActivationDerivative)

        for currentLayerNum in range(len(self.layers)-2,0,-1): #Starts at second last layer goes to second layer
            currentLayer = activationLayers[curretLayerNum]
            newErrorSignals = []
            for neuronIndex in range(len(currentLayer)): #Current Neuron being trained
                nonlinDerivative = NeuralNetwork.nonlin(currentLayer[neuronIndex],derivative=True)
                errorDerivative = self.hiddenErrorDerivative(neuronIndex,currentLayerNum,errorSignals)
                errorSignal = nonlinDerivative * errorDerivative
                newErrorSignals.append(errorSignal)
                for weightIndex in range(len(activationLayers[curretLayerNum - 1])): #Current Weight in Neuron being trained
                    previousActivationDerivative = activationLayers[curretLayerNum - 1][weightIndex]
                    self.backPropogation(neuronIndex,weightIndex,curretLayerNum,errorSignal,previousActivationDerivative)
            errorSignals = newErrorSignals
            
    def backPropogation(self,neuronIndex,weightIndex,curretLayerNum,errorSignal,previousActivationDerivative):
        """
        Basic Chain Rule step of finding derivative ie direction to move
        Gradient Descent
        """
        neuronWeightDerivative = errorSignal * previousActivationDerivative #Derivative
        delta = neuronWeightDerivative * self.learningRate #Delta Change
        if (curretLayerNum,neuronIndex,weightIndex) in self.delta:
            self.delta[(curretLayerNum,neuronIndex,weightIndex)] += delta
        else:
            self.delta[(curretLayerNum,neuronIndex,weightIndex)] = delta

    def hiddenErrorDerivative(self,neuronIndex,currentLayerNum,errorSignals):
        """
        The sum of all errors that contributed by a given weight
        """
        totalErrorDerivative = 0 #error that given weight contributes
        #Goes through the next layer in errors in their neurons
        nextLayer = self.layers[currentLayerNum+1]
        for nextLayerIndex in range(len(nextLayer.neurons)):
            weightDerivative = nextLayer.neurons[nextLayerIndex].weights[neuronIndex] #Weight that goes towards Neuron
            errorSignal = errorSignals[nextLayerIndex] #current neurons error contribution
            totalErrorDerivative += weightDerivative * errorSignal
        return totalErrorDerivative

    @staticmethod
    def calculateTotalError(actualOutputs,guessedOutputs):
        return sum(NeuralNetwork.calculateError(target,guess) for target,guess in zip(actualOutputs,guessedOutputs))

    @staticmethod
    def calculateError(actual,guess):
        return 0.5 * (actual-guess)**2

    @staticmethod
    def errorDerivative(target,guess):
        return (target - guess)

    def getError(self,inputs,actual):
        guessed = self.feedForward(inputs)[-1]
        totalError = self.calculateTotalError(actual,guessed)
        return totalError

    def display(self):
        for layer in self.layers:
            print("Layer Num: " + str(layer.layerNum))
            layer.display()
    
    def applyDelta(self):
        for indexes in self.delta:
            curretLayerNum,neuronIndex,weightIndex = indexes
            delta = self.delta[indexes]
            self.layers[curretLayerNum].neurons[neuronIndex].weights[weightIndex] -= delta
            self.delta[indexes] = 0

    @staticmethod
    def createNN(fileName):
        '''
        Creates a Neural Network with the given inputs and outputs
        '''
        network = []
        with open(fileName) as f:
            size = int(f.readline().strip().split()[1])
            for i in range(size):
                layerSize = int(f.readline().strip())
                network.append(layerSize)
            NN = NeuralNetwork(network)
            for i,layerSize in enumerate(network[1:]):
                layerNeurons = NN.layers[i+1].neurons
                for neuron in layerNeurons:
                    neuron.weights = [float(i) for i in f.readline().strip().split()]
                    neuron.biases = [float(i) for i in f.readline().strip().split()]
        return NN

    def createNNFile(self,fileName):
        file = open(fileName,'w')
        file.write("Network " + str(len(self.layerInformation))+'\n')
        for layerSize in self.layerInformation:
            file.write(str(layerSize)+'\n')
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                file.write(' '.join([str(weight) for weight in neuron.weights]) +'\n')
                file.write(' '.join([str(bias) for bias in neuron.biases])+'\n')
        file.close()

class Neuron:
    """
    Holds weights and biases
    """
    def __init__(self,layerNum,index,previousLayerSize):
        #Should Have multiple weights
        self.layerNum = layerNum
        self.index = index
        self.previousLayerSize = previousLayerSize
        self.weightsAndBiases(previousLayerSize)

    def weightsAndBiases(self,previousLayerSize):
        #If previous layer size
        if previousLayerSize == None:
            self.weights = None
            self.biases = None
        else:
            self.weights = [randomValue(-1,1) for connection in range(previousLayerSize)]
            self.biases = [0 for connection in range(previousLayerSize)]#[randomValue(-1,1) for connection in range(previousLayerSize)]

    def getActivation(self,activation,neuronIndex):
        activation = (self.weights[neuronIndex] * activation) + self.biases[neuronIndex]
        nonlinActivation = NeuralNetwork.nonlin(activation)
        return nonlinActivation

    def display(self):
        print("W: {} B: {}".format(self.weights,self.biases))
        
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
            #Passes all previous Activations into current Neurons weights
            netActivation = sum(currentNeuron.getActivation(previousLayerActivation,neuronIndex) for neuronIndex,previousLayerActivation in enumerate(activations))
            averagedActivation = netActivation / len(activations)
            newActivations.append(averagedActivation)
        return newActivations
    
    def display(self):
        for neuron in self.neurons:
            neuron.display()

    def getPreviousLayerSize(self):
        if self.layerNum > 0: return self.NN.layerInformation[self.layerNum - 1]
        else: return None

if __name__ == '__main__':
    test = False
    if test:
        NN = NeuralNetwork([1,3,3,1])
        trainingInputs = [[1],[0]]
        trainingOutputs = [[1],[0]]
        iterations = 50
        for inputs in trainingInputs:
            print('Initial',NN.feedForward(trainingInputs[0])[-1])

        for i in range(iterations):
            for inputs,outputs in zip(trainingInputs,trainingOutputs):
                NN.train(inputs,outputs)
            if i % 10:
                NN.applyDelta()

        for inputs in trainingInputs:
            print('Final',NN.feedForward(trainingInputs[0])[-1])
    else:
        NN = NeuralNetwork.createNN('networkInformation.txt')
        inputs = [0.5,0.1]
        outputs = [0.01,0.99]
        iterations = 1000
        print('Initial Results',NN.feedForward(inputs)[-1])

        for i in range(iterations):
            NN.train(inputs,outputs)
            NN.applyDelta()

        print('Final Results',NN.feedForward(inputs)[-1])
        # NN.createNNFile("trainedNeuralNetwork.txt") #Creates a file using NN features
else:
    print("Not in Main")
