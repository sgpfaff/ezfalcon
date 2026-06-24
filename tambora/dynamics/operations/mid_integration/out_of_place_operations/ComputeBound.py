from .OutOfPlaceOperation import OutOfPlaceOperation

class ComputeBound(OutOfPlaceOperation):
    '''
    Operation to compute and store particle boundedness and 
    stripping times during integration.
    '''
    def operation(self, pos, vel, mass, t): ...