from ..Operation import Operation
from abc import abstractmethod

class MidIntegrationOperation(Operation):
    '''
    Operation on particle pos, vel, mass that occurs between
    integration steps.
    '''
    @abstractmethod
    def operation(pos, vel, mass, t): ...