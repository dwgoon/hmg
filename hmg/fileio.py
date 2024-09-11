import importlib

import numpy as np
import pandas as pd    
import bitstring

import hmg.graph


class FileIO(object):
    def __init__(self, engine):
        self._engine = engine
    
    @property
    def engine(self):
        return self._engine
        
    def read_msg(self, fpath):
        nums = np.fromfile(fpath, dtype='uint8')
        return bitstring.BitArray(nums.tobytes())    
            
    def write_msg(self, fpath, msg_bits):
        msg_bytes = msg_bits.tobytes()
        with open(fpath, "wb") as fout:
            fout.write(msg_bytes)

    def from_pandas_dataframe(self, df, directed):
        raise NotImplementedError()
        
    def read_edgelist(self, fpath, directed=False, nrows=None):
        df = pd.read_csv(fpath, dtype=str, nrows=nrows)
        columns = list(df.columns)
        columns[0] = "Source"
        columns[1] = "Target"
        df.columns = columns
        g = self.from_pandas_dataframe(df.drop_duplicates(), directed)
        return g, df
        
    def write_edgelist(self, fpath, df):
        df.to_csv(fpath, index=False)
        
    def read_ogb(self, fpath, directed=False, nrows=None):
        return self.read_edgelist(fpath, directed, nrows)
        
    def write_ogb(self, fpath, df):
        self.write_edgelist(fpath, df)
    
    def read_sif(self, fpath, directed=False, header=None, nrows=None):
        df = pd.read_csv(fpath, delim_whitespace=True, header=header, nrows=nrows, dtype=str)
        if header:
            columns = df.columns.copy().tolist()
            columns[2], columns[1] = columns[1], columns[2]
        else:        
            df = df[[0, 2, 1]]
            columns = ['Source', 'Target', 'Relationship']
            df.columns = columns
        
        g = self.from_pandas_dataframe(df.drop_duplicates(), directed)
        
        return g, df  
        
    def write_sif(self, fpath, df, header=None, columns=None):
        if not columns:
            columns = df.columns.tolist()
            columns[2], columns[1] = columns[1], columns[2]
            df = df[columns]    
            
        df.to_csv(fpath, sep=' ', index=False, header=header, columns=columns)   
    
    def read_database_string(self, fpath, directed=False, nrows=None):
        df = pd.read_csv(fpath, delim_whitespace=True, dtype=str, nrows=nrows)
        g = self.from_pandas_dataframe(df.drop_duplicates(), directed)
        
        return g, df
    
    def write_database_string(self, fpath, df, columns=None):
        if columns:
            df.to_csv(fpath, sep=' ', index=False, columns=columns)   
        else:
            df.to_csv(fpath, sep=' ', index=False)
            

class NetworkxIO(FileIO):
    def __init__(self, engine):
        super().__init__(engine)
        self.nx = importlib.import_module("networkx")    

    def from_pandas_dataframe(self, df, directed):
        
        edge_attr = df.columns[2:].tolist()
        if len(edge_attr) == 0:
            edge_attr = None
            
        if directed:
            create_using = self.nx.DiGraph
        else:
            create_using = self.nx.Graph

        g = self.nx.from_pandas_edgelist(df,
                                         source=df.columns[0],
                                         target=df.columns[1],
                                         edge_attr=edge_attr,
                                         create_using=create_using)


        return self.engine.create_graph(g)
    
    
class IgraphIO(FileIO):
    def __init__(self, engine):
        super().__init__(engine)
        self.igraph = importlib.import_module("igraph")    
            
    def from_pandas_dataframe(self, df, directed=False):
            
        g = self.igraph.Graph.DataFrame(df, directed=directed)

        return self.engine.create_graph(g)
