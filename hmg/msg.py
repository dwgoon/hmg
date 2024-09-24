import numpy as np
import bitstring


def generate_bits(n_bytes, encryption=False):
    """Generate bits using bitstring.BitArray object.
    
    n_bytes: int
        Number of data points in bytes.
        
    """
    if not encryption:
        nums = np.random.randint(0, 256,
                                 size=(n_bytes,),
                                 dtype='uint8')
        data = bitstring.BitArray(nums.tobytes())    
        return data
    
    from Crypto.Cipher import AES

    key   = bytes([0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF])
    aad   = bytes([0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E]) 
    nonce = bytes([0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB]) # 7 ~ 13Byte
    
    nums = np.random.randint(0, 256,
                             size=(n_bytes,),
                             dtype='uint8')
    

    cipher = AES.new(key, AES.MODE_CCM, nonce)
    cipher.update(aad)
    cipher_data = cipher.encrypt(nums.tobytes())        
    arr_cipher = np.frombuffer(cipher_data, dtype=np.uint8)
    return bitstring.BitArray(arr_cipher.tobytes())    
    
def generate_bytes(n_bytes):
    """Generate bytes using numpy.NDArray object.
    
    n_bytes: int
        Number of data points in bytes.
        
    """
    # return np.random.bytes(n_bytes)

    return np.random.randint(0, 256,
                             size=(n_bytes,),
                             dtype='uint8')
    

def to_bitarr(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr should be numpy.ndarray.")
        
    return bitstring.BitArray(arr)    
    
if __name__ == "__main__":
    from Crypto.Cipher import AES
    
    def print_hex_bytes(name, byte_array):
        print('{} len[{}]: '.format(name, len(byte_array)), end='')
        for idx, c in enumerate(byte_array):
            print("{}".format(hex(c)), end='')
            if idx < len(byte_array)-1:
                print(',', end='')
        print("")
    
    # 각종 키 정보
    key   = bytes([0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF])
    aad   = bytes([0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E]) 
    nonce = bytes([0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB]) # 7 ~ 13Byte
    
    # 암호화된 데이터
    data  = bytes([0x88,0x3f,0x4f,0x94,0xae,0x6c,0xda,0x41,0xa7,0xdd,0x35])
    # MAC TAG
    mac = bytes([0x1c,0xad,0x41,0xac,0xe1,0x76,0x9,0xd1,0x28,0xe7,0x89,0x24,0x39,0x2,0x8,0xba])
    
    
    # 각각의 키 출력
    print_hex_bytes('key', key)
    print_hex_bytes('aad', aad)
    print_hex_bytes('nonce', nonce)
    
    # 암호화 라이브러리 생성
    cipher = AES.new(key, AES.MODE_CCM, nonce)
    # aad(Associated Data) 추가
    cipher.update(aad)
    cipher_data = cipher.encrypt(data)
    
    
    # try:
    #     # 복호화!!!
    #     # cipher.update(aad)
    #     # plain_data = cipher.decrypt_and_verify(cipher_data, mac)
    #     print('---------------------------------')
    #     # 암호화된 데이터 출력
    #     print_hex_bytes('plain_data', plain_data)
    
    # except ValueError:
    #     # MAC Tag가 틀리다면, 즉, 훼손된 데이터
    #     print ("Key incorrect")
    # #[출처] 파이썬(Python) - AES-CCM 암호화 복호화 예제 by pycryptodome|작성자 천동이