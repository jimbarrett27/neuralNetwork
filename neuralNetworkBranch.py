import numpy as np

class NeuralNetworkBranch(object):

	def __init__(self,leftUnit,rightUnit):
		
#		self.weight = self.randomWeight()
		self.weight = np.random.randn()*0.01
		
		self._leftUnit = leftUnit
		self._rightUnit = rightUnit

	@property
	def leftUnit(self):
		"""The unit on the left of this branch

		"""

		return self._leftUnit

	@property
	def rightUnit(self):
		"""The unit on the right of this branch

		"""

		return self._rightUnit

	
	
			
