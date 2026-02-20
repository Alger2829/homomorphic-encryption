import numpy as np
from numpy.polynomial import polynomial as poly

def polymul_fhe(x, y, modulus, poly_mod):
  #polynomial multiplication with modulus
    return np.int64(
        np.round(poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus)
    )


def polyadd_fhe(x, y, modulus, poly_mod):
  #polynomial multiplication with modulus
    return np.int64(
        np.round(poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus)
    )

#returns array of coefficients with the coeff[i] being the coeff of x ^ i.
def gen_binary_poly_fhe(size):
    #generates polynomial with coefficients in [0,1]
    return np.random.randint(0, 2, size, dtype=np.int64)


def gen_uniform_poly_fhe(size, modulus):
    #Generates a polynomial with coeffecients being integers in Z_modulus
    return np.random.randint(0, modulus, size, dtype=np.int64)


def gen_normal_poly_fhe(size):
   #Generates a polynomial with coeffecients in a normal distribution of mean 0 and a standard deviation of 2, then discretize it.
    return np.int64(np.random.normal(0, 2, size=size))


def keygen_fhe(size, modulus, poly_mod):
    #Generate a public and secret keys
    sk = gen_binary_poly_fhe(size)
    a = gen_uniform_poly_fhe(size, modulus)
    e = gen_normal_poly_fhe(size)
    b = polyadd_fhe(polymul_fhe(-a, sk, modulus, poly_mod), -e, modulus, poly_mod)
    return (b, a), sk
    #secret key - sk, public key - (b,a)

# def keygen_FVSH(n, p):
#   """Args:
#       n: polynomial degree.
#       p: prime number for the ciphertext modulus.

#   Returns:
#       Public key (rlk) as a random polynomial of degree n-1 modulo p.
#   """
#   # Sample a random polynomial of degree n-1 modulo p
#   rlk = np.random.randint(0, p, n, dtype=np.int64)[:n-1]  # Ensure degree n-1
#   return rlk

# Scheme's parameters
# polynomial modulus degree
n = 2**4
# ciphertext modulus
q = 2**15
# plaintext modulus
t = 2**8
# polynomial modulus
poly_mod = np.array([1] + [0] * (n - 1) + [1])
# Keygen
pk, sk = keygen_fhe(n, q, poly_mod)


def encrypt_fhe(pk, pt):
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (n - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m  % q
    e1 = gen_normal_poly_fhe(n)
    e2 = gen_normal_poly_fhe(n)
    u = gen_binary_poly_fhe(n)
    ct0 = polyadd_fhe(
            polyadd_fhe(
                polymul_fhe(pk[0], u, q, poly_mod),
                e1, q, poly_mod),
            scaled_m, q, poly_mod
        )
    ct1 = polyadd_fhe(
            polymul_fhe(pk[1], u, q, poly_mod),
            e2, q, poly_mod
        )
    return (ct0, ct1)

def decrypt_fhe(sk, ct):
    scaled_pt = polyadd_fhe(
            polymul_fhe(ct[1], sk, q, poly_mod),
            ct[0], q, poly_mod
        )
    decrypted_poly = np.round(scaled_pt * t / q) % t
    return int(decrypted_poly[0])

def add_plain_fhe(ct, pt):
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m  % q
    new_ct0 = polyadd_fhe(ct[0], scaled_m, q, poly_mod)
    return (new_ct0, ct[1])


def mul_plain_fhe(ct, pt):
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    new_c0 = polymul_fhe(ct[0], m, q, poly_mod)
    new_c1 = polymul_fhe(ct[1], m, q, poly_mod)
    return (new_c0, new_c1)

def add_cipher_fhe(ct1, ct2):
    # Ensure the ciphertexts have the same length
    assert len(ct1) == len(ct2), "Ciphertexts must have the same length"

    # Add the elements of the ciphertext tuples element-wise
    new_ct0 = polyadd_fhe(ct1[0], ct2[0], q, poly_mod)
    new_ct1 = polyadd_fhe(ct1[1], ct2[1], q, poly_mod)
    return (new_ct0, new_ct1)

def mul_cipher_fhe(ct1, ct2):
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([ct2] + [0] * (size - 1), dtype=np.int64) % t
    new_c0 = polymul_fhe(ct1[0], m, q, poly_mod)
    new_c1 = polymul_fhe(ct1[1], m, q, poly_mod)
    return (new_c0, new_c1)











