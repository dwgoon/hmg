import importlib


class Graph(object):    
    def __init__(self, g):
        self._graph = g
    
    def degree(self, x):
        raise NotImplementedError()
    
    def num_nodes(self):
        raise NotImplementedError()
        
    def num_edges(self):
        raise NotImplementedError()
    
    def is_directed(self):
        raise NotImplementedError()
    
    def has_node(self, x):
        raise NotImplementedError()
    
    def has_edge(self, x, y):
        raise NotImplementedError()
    
    def add_edge(self, x, y):
        raise NotImplementedError()    
    
    @property
    def graph(self):
        return self._graph


class NetworkxGraph(Graph):    
    def __init__(self, g):                
        super().__init__(g)
        self.nx = importlib.import_module("networkx")    
    
    def __eq__(self, g):
        return self.nx.is_isomorphic(self._graph, g._graph)
    
    def degree(self, x=None):
        if x:
            return self._graph.degree(x)
        else:
            return self._graph.degree
    
    def num_nodes(self):
        return self._graph.number_of_nodes()
    
    def num_edges(self):
        return  self._graph.number_of_edges()
    
    def is_directed(self):
        return self._graph.is_directed()
    
    def has_node(self, x):
        return self._graph.had_node(x)
    
    def has_edge(self, x, y):
        return self._graph.has_edge(x, y)
    
    def add_edge(self, x, y):
        return self._graph.add_edge(x, y)
    
    
class IgraphGraph(Graph):    
    def __init__(self, g):                
        super().__init__(g)
        self.igraph = importlib.import_module("igraph")    
        
    def __eq__(self, g):
        return self._graph.isomorphic(g._graph)

    def degree(self, x):
        if x:
            return self._graph.degree(x)
        else:
            return self._graph.degree
        
    def num_nodes(self):
        return self._graph.vcount()
    
    def num_edges(self):
        return  self._graph.ecount()
    
    def is_directed(self):
        return self._graph.is_directed()
    
    def has_node(self, x):
        try:
            self._graph.vs.find(name=x)                    
        except Exception:
            return False
        
        return True
            
    def has_edge(self, x, y):
        try:
            res = self._graph.get_eid(x, y)
            if res < 0:
                return False
        except Exception:
            return False
        
        return True
                
    def add_edge(self, x, y):        
        try:
            ix_x = self._graph.vs.find(name=x).index                           
        except Exception:
            self._graph.add_vertex(x)
            ix_x = self._graph.vs.find(name=x).index                           

        try:
            ix_y = self._graph.vs.find(name=y).index                            
        except Exception:
            self._graph.add_vertex(y)     
            ix_y = self._graph.vs.find(name=y).index                            

        self._graph.add_edge(ix_x, ix_y)