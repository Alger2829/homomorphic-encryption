
import math
import random

def is_prime_rsa(number):
    if number < 2:
        return False
    for i in range(2, number // 2 + 1):
        if number % i == 0:
            return False
    return True

def generate_prime_rsa(min_value, max_value):
    prime = random.randint(min_value, max_value)
    while not is_prime_rsa(prime):
        prime = random.randint(min_value, max_value)
    return prime

def mod_inverse_rsa(e, phi):
    for d in range(3, phi):
        if (d * e) % phi == 1:
            return d
    raise ValueError("mod inverse doesn't exist")

def keygen_rsa():
    # select random large prime integers: p, q such that p != q
    p, q = generate_prime_rsa(1000, 5000), generate_prime_rsa(1000, 5000)
    while p == q:
        q = generate_prime_rsa(1000, 5000)

    # compute n = p * q
    n = p * q
    # phi(n) = (p-1) * (q-1)
    phi_n = (p - 1) * (q - 1)

    # assume e such that gcd (e, phi(n)) = 1 & 1 < e < phi(n)
    e = random.randint(3, phi_n - 1)
    while math.gcd(e, phi_n) != 1:
        e = random.randint(3, phi_n - 1)

    # find d such that d * e mod phi(n) = 1
    d = mod_inverse_rsa(e, phi_n)

    # private key = (d, n)
    private_key = (d, n)

    # public key = (e, n)
    public_key = (e, n)
    return (private_key, public_key)

def encrypt_rsa(message, public_key):
    e, n = public_key
    m = message
    # ciphertext = m^e mod n
    ciphertext = pow(m, e, n)
    return ciphertext

def decrypt_rsa(ciphertext, private_key):
    d, n = private_key
    # decrypted message = c^d mod n
    dec_message = pow(ciphertext, d, n)
    return dec_message

def decrypt_product_rsa(encrypted_product, private_key):
    # Decrypt the product
    decrypted_product = decrypt_rsa(encrypted_product, private_key)
    return decrypted_product
