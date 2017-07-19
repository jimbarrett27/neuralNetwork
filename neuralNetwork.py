import numpy as np
import copy
import scipy.optimize as spop
import warnings

import neuralNetworkUnit as nnu
import neuralNetworkBranch as nnb

class NeuralNetwork(object):

	def __init__(self, nInputs, nOutputs, hiddenLayersDims, outputActivationFunctions = None, outputActivationDerivatives = None, hiddenActivationFunctions = None,\
				 hiddenActivationDerivatives = None):
		"""Constructs a NeuralNetwork object. Input parameters shoudl be

			nInputs - The number of inputs to the model

			nOutputs - The number of outputs to the model

			hiddenLayersDims - a 2-tuple of the form (nHiddenLayers,nUnitsPerLayer), where nUnitsPerLayer excludes the hidden bias unit

			outputActivationFunctions - an array of length nOutputs containing activation functions for the outputs
			outputActivationDerivatives - an array of length nOutputs containing the first derivatives of the activation functions for the outputs

			hiddenActivationFunctions - an array of shape (nHiddenLayers,nUnitsPerLayer) containing activation functions for the hidden units
			hiddenActivationDerivatives - an array of shape (nHiddenLayers,nUnitsPerLayer) containing the first derivatives of the activation 				functions for the hidden units

		"""  

		self._nInputs = nInputs
		self._nOutputs = nOutputs

		self._nHiddenLayers, self._nUnitsPerLayer = hiddenLayersDims

		self._outputActivationFunctions = outputActivationFunctions
		self._outputActivationDerivatives = outputActivationDerivatives

		self._hiddenActivationFunctions = hiddenActivationFunctions
		self._hiddenActivationDerivatives = hiddenActivationDerivatives

		self.initialiseActivationFunctions()

		self.initialiseNetwork()

		self._nBranches = len(self.collectAllBranches())
		
	@property
	def nInputs(self):
		"""The number of inputs to the model
	
		"""

		return self._nInputs
	
	@property
	def nOutputs(self):
		"""The number of outputs to the model

		"""

		return self._nOutputs
	
	@property
	def nHiddenLayers(self):
		"""The number of hidden layers


		"""

		return self._nHiddenLayers

	@property
	def nUnitsPerLayer(self):
		"""The number of hidden units on each of the hidden layers

		"""

		return self._nUnitsPerLayer

	@property
	def outputActivationFunctions(self):
		"""The activation functions for the output layer

		"""

		return self._outputActivationFunctions

	@property
	def outputActivationDerivatives(self):
		"""The first derivatives of the activation functions for the output layer

		"""

		return self._outputActivationDerivatives

	@property
	def hiddenActivationFunctions(self):
		"""The activation functions for the hidden layers

		"""

		return self._hiddenActivationFunctions

	@property
	def hiddenActivationDerivatives(self):
		"""The first derivatives of the activation functions for the hidden layers

		"""

		return self._hiddenActivationDerivatives

	@property
	def nBranches(self):
		"""The number of branches in the network

		"""

		return self._nBranches
	
	def initialiseActivationFunctions(self):
		"""Set the activation functions to common defaults if we haven't been given any explicitly

		"""

		###uniform for output units
		if self._outputActivationFunctions == None or self._outputActivationDerivatives == None:	
	
			self._outputActivationFunctions = []
			self._outputActivationDerivatives = []

			actFunc = lambda x : x
			dActFunc = lambda x : 1.0
	
			for i in range(self.nOutputs):
				
				self._outputActivationFunctions.append(actFunc)
				self._outputActivationDerivatives.append(dActFunc)

			self._outputActivationFunctions = np.array(self._outputActivationFunctions)
			self._outputActivationDerivatives = np.array(self._outputActivationDerivatives)
			

		if self._hiddenActivationFunctions == None or self._hiddenActivationDerivatives == None:

			self._hiddenActivationFunctions = []
			self._hiddenActivationDerivatives = []

			for i in range(self.nHiddenLayers):

				fTemp = []
				dTemp = []
				
				#Make the default sigmoid the one suggested in LeCun et al 1998
				twist = 0.01
				a = 1.7159
				c = 2.0/3.0

				actFunc = lambda x : a*np.tanh(c*x) + twist*x
				dActFunc = lambda x : twist + a*c*(1.0 - (np.tanh(c*x)**2.0))

#				actFunc = lambda x : np.tanh(x)
#				dActFunc = lambda x : 1.0 - np.tanh(x)**2.0

				#plus all of the bias
				for j in range(self.nUnitsPerLayer+1):
					
					fTemp.append(actFunc)
					dTemp.append(dActFunc)
				
				self._hiddenActivationFunctions.append(fTemp)
				self._hiddenActivationDerivatives.append(dTemp)
			
			self._hiddenActivationFunctions = np.array(self._hiddenActivationFunctions)
			self._hiddenActivationDerivatives = np.array(self._hiddenActivationDerivatives)

	def initialiseNetwork(self):

		self.initialiseInputLayer()
		self.initialiseHiddenLayers()
		self.initialiseOutputLayer()
			
	def initialiseInputLayer(self):

		#make nInputs input units, plus 1 for the bias
		self.inputLayer = []
		unit = nnu.NeuralNetworkUnit('bias',lambda x:x,lambda x:1.0)
		self.inputLayer.append(unit)
		for i in range(self.nInputs):
			unit = nnu.NeuralNetworkUnit('input',lambda x:x,lambda x:1.0)
			self.inputLayer.append(unit)
	
		self.inputLayer = np.array(self.inputLayer)


	def initialiseHiddenLayers(self):

		#make all of the hidden layers of the Neural Network, with the zero'th element of each layer as the bias
		self.hiddenLayers = []
		
		#make first bias, and connect the first hidden layer to the input layer
		layer = []
		biasUnit = nnu.NeuralNetworkUnit('bias',lambda x:x,lambda x:1.0)
		layer.append(biasUnit)
		for i in range(self.nUnitsPerLayer):
			
			unit = nnu.NeuralNetworkUnit('hidden',self.hiddenActivationFunctions[0,i+1],self.hiddenActivationDerivatives[0,i+1])
			
			for inUnit in self.inputLayer:
				branch = nnb.NeuralNetworkBranch(inUnit,unit)
				unit.attachBranchIn(branch)
				inUnit.attachBranchOut(branch)

			layer.append(unit)

		self.hiddenLayers.append(layer)

		#now make the rest of the hidden layers, connecting them to the previous one
		for i in range(1,self.nHiddenLayers):
			
			lastLayer = self.hiddenLayers[-1]
			newLayer = []
			biasUnit = nnu.NeuralNetworkUnit('bias',self.hiddenActivationFunctions[i,0],self.hiddenActivationDerivatives[i,0])
			newLayer.append(biasUnit)

			for j in range(self.nUnitsPerLayer):
				unit = nnu.NeuralNetworkUnit('hidden',self.hiddenActivationFunctions[i,j+1],self.hiddenActivationDerivatives[i,j+1])

				for lastUnit in lastLayer:
					branch = nnb.NeuralNetworkBranch(lastUnit,unit)
					unit.attachBranchIn(branch)
					lastUnit.attachBranchOut(branch)

				newLayer.append(unit)
			
			self.hiddenLayers.append(newLayer)

		self.hiddenLayers = np.array(self.hiddenLayers)


	def initialiseOutputLayer(self):

		#finally, make the output layer, connecting it to the last hidden layer. This one doesn't have a bias
		self.outputLayer = []
		lastHiddenLayer = self.hiddenLayers[-1]
		for i in range(self.nOutputs):

			unit = nnu.NeuralNetworkUnit('output',self.outputActivationFunctions[i],self.outputActivationDerivatives[i])		
			
			for hiddenUnit in lastHiddenLayer:
				branch = nnb.NeuralNetworkBranch(hiddenUnit,unit)
				unit.attachBranchIn(branch)
				hiddenUnit.attachBranchOut(branch)
			
			self.outputLayer.append(unit)
	
		self.outputLayer = np.array(self.outputLayer)

	def collectAllBranches(self):
		"""Helper method to collect references to all branches in a standard way, so that we can work out their 
		derivatives and update the weights.

		Reads through network units from top to bottom, left to right, taking the branches to the right in the order they're stored

		"""
		allBranches = []

		for unit in self.inputLayer:

			for branch in unit.branchesOut:

				allBranches.append(branch)

		for layer in self.hiddenLayers:

			for unit in layer:
			
				for branch in unit.branchesOut:

					allBranches.append(branch)

		for unit in self.outputLayer:

			for branch in unit.branchesOut:

				allBranches.append(branch)


		return allBranches

	def updateWeights(self,weightUpdate):
		"""Given a set of weight updates for the branches calculated in the order in collectAllBranches(), update the weights!

		"""
	
		branches =  self.collectAllBranches()

		for i in range(self.nBranches):

			branches[i].weight -= weightUpdate[i]

	def reassignWeights(self,weights):
		"""Given a set of weights for the branches calculated in the order in collectAllBranches(), change the weights on all branches (useful for numerical checking)

		"""
	
		branches =  self.collectAllBranches()

		for i in range(self.nBranches):

			branches[i].weight = weights[i]

	def feedForward(self, inputs):
		"""Propagate the inputs through the network

		"""

		inputs = np.atleast_1d(inputs)

		if not len(inputs) == self.nInputs:

			raise ValueError("The input vector is the wrong length for this network")

		#don't forget we have a bias unit in here too
		for i in range(1,self.nInputs+1):
			self.inputLayer[i].activation = inputs[i-1]
			self.inputLayer[i].output = inputs[i-1]			

		for layer in self.hiddenLayers:

			for unit in layer:

				unit.forwardValue()

		for unit in self.outputLayer:
	
			unit.forwardValue()

	def copy(self):

		return copy.deepcopy(self)

	def getWeights(self):

		branches = self.collectAllBranches()

		weights = []
		for b in branches:

			weights.append(b.weight)

		return np.atleast_1d(weights)
		
	
	def __call__(self,inputs):

		self.feedForward(inputs)

		outputs = []

		for unit in self.outputLayer:

			outputs.append(unit.output)

		return np.atleast_1d(outputs)

	def makeFastFeedForwardFunction(self):
		"""A faster, vectorisable way of feeding forward. DOESNT WORK FOR DIFFERENT ACTIVATION FUNCTIONS

		"""

		outWeightMatrix = []
		for unit in self.outputLayer:

			row = []
			for b in unit.branchesIn:
				print b.weight
				row.append(b.weight)
			
			outWeightMatrix.append(row)
		outWeightMatrix = np.array(outWeightMatrix).squeeze()

		hiddenMatrices = []
		for layer in self.hiddenLayers:
			matrix = []
			#ignore the bias unit, since it has no branches in
			for unit in layer[1:]:
				row = []
				for b in unit.branchesIn:
					row.append(b.weight)

				matrix.append(row)
			matrix = np.array(matrix)

			hiddenMatrices.append(matrix)

		hidActFunc = (self.hiddenLayers[0])[1].activationFunction
		outActFunc = self.outputLayer[0].activationFunction

		def ffFunc(inp):
	
			forward = np.insert(inp.T,0,1.0,axis=0)
			for matrix in hiddenMatrices:
				next = np.dot(matrix,forward)
				next = hidActFunc(next)
				forward = np.insert(next,0,1.0,axis=0)

			out = np.dot(outWeightMatrix,forward)

			return outActFunc(out)

		return ffFunc


	def printNetwork(self):
		"""Print all of the information about the network, for debugging


		"""
		i = 0
		for branch in self.collectAllBranches():
			
			print 'branch', i
			print 'weight', branch.weight

			lUnit = branch.leftUnit
			rUnit = branch.rightUnit
			
			print 'left unit type/activation/output/delta', lUnit.unitType, lUnit.activation, lUnit.output, lUnit.delta
			print 'right unit type/activation/output/delta', rUnit.unitType, rUnit.activation, rUnit.output, rUnit.delta

			print '\n'
			
			i+=1

		

		
		

	

			

		


	

	

			

		

		
		

			
		
		
		
		

		


	

		

	

		

		

			
