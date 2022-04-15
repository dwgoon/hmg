import numpy as np
import bitstring


def generate_bits(n_bytes):
    """Generate bits using bitstring.BitArray object.
    
    n_bytes: int
        Number of data points in bytes.
        
    """
    nums = np.random.randint(0, 256,
                             size=(n_bytes,),
                             dtype='uint8')
    data = bitstring.BitArray(nums.tobytes())    
    return data

def generate_bytes(n_bytes):
    """Generate bytes using numpy.NDArray object.
    
    n_bytes: int
        Number of data points in bytes.
        
    """
    # return np.random.bytes(n_bytes)

    return np.random.randint(0, 256,
                             size=(n_bytes,),
                             dtype='uint8')
    

