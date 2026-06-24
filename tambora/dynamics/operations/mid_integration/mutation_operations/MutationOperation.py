from ..MidIntegrationOperation import MidIntegrationOperation
from typing import Tuple
from abc import abstractmethod

class MutationOperation(MidIntegrationOperation):
    '''
    Mid-integration operation on particle ensemble that returns
    new particles.
    '''
    @abstractmethod
    def operation(self, pos, vel, mass, t) -> Tuple: ...