import importlib


class Graph(object):    
    def __init__(self, g):
        self._graph = g
        
    def copy(self):
        raise NotImplementedError()
        
    def to_directed(self):
        raise NotImplementedError()
    
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
    
    def has_edge(self, x, y, directed=True):
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
            
    def copy(self):
        return self.__class__(self._graph.copy())
    
    def to_directed(self):
        return self.__class__(self._graph.to_directed())
        
    def __eq__(self, g):
        return self.nx.is_isomorphic(self._graph, g._graph)
    
    def degree(self, x=None):
        if x is not None:
            return self._graph.degree(x)
        else:
            return self._graph.degree

    def neighbors(self, x):
        return self._graph.neighbors(x)

    def successors(self, x):
        return self._graph.successors(x)

    def predecessors(self, x):
        return self._graph.predecessors(x)

    def num_nodes(self):
        return self._graph.number_of_nodes()
    
    def num_edges(self):
        return  self._graph.number_of_edges()
    
    def is_directed(self):
        return self._graph.is_directed()
    
    def has_node(self, x):
        return self._graph.had_node(x)
    
    def has_edge(self, x, y, directed=True):
        if directed:
            return self._graph.has_edge(x, y)
        else:
            return self._graph.has_edge(x, y) or self._graph.has_edge(y, x)
    
    def add_edge(self, x, y, *args, **kwargs):
        return self._graph.add_edge(x, y, *args, **kwargs)
    
    def del_edge(self, x, y):
        return self._graph.remove_edge(x, y)
    
    def add_node(self, x):
        return self._graph.add_node(x)

    def del_node(self, x):
        return self._graph.remove_node(x)

    
class IgraphGraph(Graph):    
    def __init__(self, g):                
        super().__init__(g)
        self.igraph = importlib.import_module("igraph")    

    def copy(self):
        return self.__class__(self._graph.copy())
    
    def to_directed(self):
        return self.__class__(self._graph.as_directed())    
        
    def __eq__(self, g):
        return self._graph.isomorphic(g._graph)

    def degree(self, x):
        if x is not None:
            return self._graph.degree(x)
        else:
            return self._graph.degree

    def neighbors(self, x):
        return self._graph.neighbors(x)

    def successors(self, x):
        return self._graph.successors(x)

    def predecessors(self, x):
        return self._graph.predecessors(x)

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
            
    def has_edge(self, x, y, directed=True):
        try:
            res_xy = self._graph.get_eid(x, y)
            if not directed:
                res_yx = self._graph.get_eid(y, x)
                if res_xy < 0  and res_yx < 0:
                    return False
            elif res_xy < 0:
                return False
        except Exception:
            return False
        
        return True
                
    def add_edge(self, x, y, *args, **kwargs):        
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

        return self._graph.add_edge(ix_x, ix_y, *args, **kwargs)
    
    def del_edge(self, x, y):
        eid = self._graph.get_eid(x, y)
        return self._graph.delete_edges(eid)
        
    def add_node(self, x):
        return self._graph.add_vertex(x)
        
    def del_node(self, x):        
        return self._graph.delete_vertices(x)