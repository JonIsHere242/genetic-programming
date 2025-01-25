from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os
import random
import pandas as pd
import base64

def generate_encryption_dataset(num_samples=500_000):
    """Simple one-shot generator for 500K samples"""
    data = {
        'plaintext': [],
        'ciphertext': [],
        'key': []
    }

    for _ in range(num_samples):
        # Generate random plaintext (16-256 bytes)
        pt = os.urandom(random.randint(16, 256))
        key = os.urandom(16)  # AES-128 key
        
        # Pad and encrypt
        padder = padding.PKCS7(128).padder()
        padded_pt = padder.update(pt) + padder.finalize()
        
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher.encryptor()
        ct = encryptor.update(padded_pt) + encryptor.finalize()
        
        # Store as base64 strings
        data['plaintext'].append(base64.b64encode(pt).decode())
        data['ciphertext'].append(base64.b64encode(ct).decode())
        data['key'].append(base64.b64encode(key).decode())

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_parquet('encryption_dataset.parquet', engine='pyarrow', compression='zstd')




if __name__ == '__main__':
    generate_encryption_dataset()
    
    # Quick verification
    df = pd.read_parquet('encryption_dataset.parquet')
    sample = df.iloc[0]
    print("First sample:")
    print(f"Plaintext: {sample['plaintext']}")
    print(f"Key: {sample['key']}")
    print(f"Ciphertext: {sample['ciphertext']}")