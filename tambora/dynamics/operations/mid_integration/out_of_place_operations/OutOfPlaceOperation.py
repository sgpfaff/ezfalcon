from ..MidIntegrationOperation import MidIntegrationOperation
from types import NoneType
from abc import abstractmethod

class OutOfPlaceOperation(MidIntegrationOperation):
    '''
    Mid-integration operation on particle ensemble that does
    not return anything.
    '''
    @abstractmethod
    def operation(self, pos, vel, mass, t) -> NoneType: ...