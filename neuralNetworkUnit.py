import numpy as np

class NeuralNetworkUnit(object):

	def __init__(self,unitType,activationFunction,activationDerivative):

		self._unitType = unitType
		self._activationFunction = activationFunction
		self._activationDerivative = activationDerivative

		self.branchesIn = []
		self.branchesOut = []

		#bias nodes always have activation unity
		if unitType == 'bias':
			self.activation = 1.0
			self.output = 1.0
		else:
			self.activation = None
			self.output = None

		self.delta = None

	@property
	def unitType(self):
		"""The type of unit this is, can take on the values:

			"input"
			"bias"
			"hidden"
			"output"

		"""

		return self._unitType

	@property
	def activationFunction(self):
		"""The 'activation function', which gets applied to the forward value

		"""

		return self._activationFunction

	@property
	def activationDerivative(self):
		"""The first derivative of the 'activation function'

		"""

		return self._activationDerivative

	def attachBranchIn(self,branchIn):
		"""Connect this unit to one in a layer to the left

		"""

		self.branchesIn.append(branchIn)

	def attachBranchOut(self,branchOut):
		"""Connect this unit to one in a layer to the right


		"""

		self.branchesOut.append(branchOut)

	def forwardValue(self):
		"""Calculates the 'forward value' for this unit, according to the formula

			u_j = h\left(\sum_{i=0}^{D} w_{ji} x_i

		"""
		
		if self.unitType == 'bias':

			self.activation = 1.0
			self.output = 1.0

		else: 
			total = 0.0
			
#			for branch in self.branchesIn:
#			
#				lUnit = branch.leftUnit
#				total += branch.weight * lUnit.output
	
			weights = [b.weight for b in self.branchesIn]
			outs = [b.leftUnit.output for b in self.branchesIn] 

			total = np.dot(weights,outs) 

			self.activation = total
			self.output = self.activationFunction(self.activation)

	def calculateDelta(self):

		#Need to explicitly set the output deltas
		if self.unitType == 'output':
		
			raise ValueError("Can't calculate delta directly on output units")
		
		total = 0.0
		for branch in self.branchesOut:

			total += branch.weight * branch.rightUnit.delta

		prefactor = self.activationDerivative(self.activation)

		self.delta = prefactor*total




















