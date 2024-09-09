
from hmg.engine import GraphEngine

class Base:    
    def __init__(self, engine=None, verbose=1):

        if engine is None:
            raise ValueError("GraphEngine is required to create an algorithm object.")

        self._engine = engine  # graph engine
        self._verbose = verbose
       
    @property
    def engine(self):
        return self._engine
    
    @property
    def verbose(self):
        return self._verbose
        
    def encode(self, g):
        raise NotImplementedError()
        
    def decode(self, g):
        raise NotImplementedError()