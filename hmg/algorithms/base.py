
from hmg.engine import GraphEngine

class Base:    
    def __init__(self, engine):
        self._engine = engine  # graph engine
       
    @property
    def engine(self):
        return self._engine
        
    def encode(self, g):
        raise NotImplementedError()
        
    def decode(self, g):
        raise NotImplementedError()