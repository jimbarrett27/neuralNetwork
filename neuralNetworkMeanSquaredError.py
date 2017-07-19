import numpy as np
import neuralNetwork as nn

class NeuralNetworkMeanSquaredError(object):
	"""Class to return the total mean squared error and derivatives of the error function with respect to the 
	weights of the neural network, and also to calculate the best step in weight-space to approach a minimum error

	"""	

	def __init__(self,trainingInputs,trainingOutputs,stepRate=0.01,printStuff=False):

		self.trainingInputs = trainingInputs.copy()
		self.trainingOutputs = trainingOutputs.copy()

		self.stepRate = stepRate

		self.printStuff = printStuff

		print self.printStuff

	def populateDeltas(self,neuralNetwork,trainIn,trainOut):

		nnOutput = neuralNetwork(trainIn)

		for i in range(neuralNetwork.nOutputs):
	
			diff = nnOutput[i] - trainOut[i]

			neuralNetwork.outputLayer[i].delta = diff

		##backwards propagate
		for layer in neuralNetwork.hiddenLayers[::-1]:

			for unit in layer:

				unit.calculateDelta()

	def calculateErrorDerivatives(self,neuralNetwork,trainIn,trainOut):

		"""According to the formulae in Bishop, this derivative is simply the product of the delta from the right unit and
			the output from the left

		"""

		neuralNetwork.feedForward(trainIn)

		self.populateDeltas(neuralNetwork,trainIn,trainOut)

		allBranches = neuralNetwork.collectAllBranches()
		derivatives = []

		for branch in allBranches:

			delta = branch.rightUnit.delta

			output = branch.leftUnit.output

			derivatives.append(delta*output)

		return np.atleast_1d(derivatives)

	def slowlyCalculateErrorDerivatives(self,neuralNetwork,epsilon=1e-6):

		cop = neuralNetwork.copy()
	
		originalWeights = neuralNetwork.getWeights()

		derivatives = []
		
		for i in range(len(originalWeights)):

			forwardWeights = originalWeights.copy()
			forwardWeights[i] += epsilon
			cop.reassignWeights(forwardWeights)
			forwardError = self.errorFunction(cop)

			backwardWeights = originalWeights.copy()
			backwardWeights[i] -= epsilon
			cop.reassignWeights(backwardWeights)
			backwardError = self.errorFunction(cop)

			derivatives.append((forwardError-backwardError)/(2.0*epsilon))

		return np.atleast_1d(derivatives)

	def calculateWeightUpdate(self,neuralNetwork):

		derivatives = np.zeros(neuralNetwork.nBranches)

		nTrain = len(self.trainingInputs)

		for i in range(nTrain):

			trainIn = self.trainingInputs[i]
			trainOut = self.trainingOutputs[i]
		
			derivatives += self.calculateErrorDerivatives(neuralNetwork,trainIn,trainOut)

		if self.printStuff:

			slowDerivatives = self.slowlyCalculateErrorDerivatives(neuralNetwork)

			print np.abs(derivatives - slowDerivatives) / np.abs(derivatives + slowDerivatives)

		weightUpdate = self.stepRate * derivatives

		return weightUpdate

	def errorFunction(self,neuralNetwork):

		errorSum = 0.0

		for i in range(len(self.trainingInputs)):

			nnOuts = neuralNetwork(self.trainingInputs[i])

			diff = nnOuts - self.trainingOutputs[i]

			errorSum += np.dot(diff,diff)

		return 0.5*errorSum

	def stochasticGradientDescent(self,neuralNetwork,iterations=1000):
			"""Use stochastic gradient descent to train the network

			"""

			for i in range(iterations):

				update = self.calculateWeightUpdate(neuralNetwork)

				neuralNetwork.updateWeights(update)

			return self.errorFunction(neuralNetwork)

	def conjugateGradient(self,neuralNetwork,iterations=100,previousDerivatives=None):

		nTrain = len(self.trainingInputs)

		##Do a normal single descent calc to kick things off
		if previousDerivatives == None:
			ind = np.random.randint(0,nTrain)
			trainIn = self.trainingInputs[ind]
			trainOut = self.trainingOutputs[ind]
			previousDerivatives = self.calculateErrorDerivatives(trainIn,trainOut)
			previousDirection = -1.0*previousDerivatives

		for i in range(iterations):

			derivatives = np.zeros(len(previousDerivatives))

			for j in range(nTrain):

				trainIn = self.trainingInputs[j]
				trainOut = self.trainingOutputs[j]

				derivative = self.calculateErrorDerivatives(trainIn,trainOut)

				derivatives += derivative

			beta = np.dot(derivatives-previousDerivatives,derivatives)/np.dot(previousDerivatives,previousDerivatives)

			newDirection = derivatives - beta*previousDirection

			neuralNetwork.updateWeights(self.stepRate*newDirection)

			previousDirection = newDirection
			previousDerivatives = derivatives

		return self.errorFunction(neuralNetwork)

				

					












