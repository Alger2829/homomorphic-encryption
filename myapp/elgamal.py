import random

def is_prime_elgamal(n):
    for i in range(2, n): # [2, n-1]
        if n % i == 0:
            return False
    return True

def gcd_elgamal(a, b):
    if a < b:
        return gcd_elgamal(b, a)
    elif a % b == 0:
        return b
    else:
        return gcd_elgamal(b, a % b)

def keygen_elgamal():
    # key = random.randint(pow(10, 20), q)
    # while gcd_elgamal(q, key) != 1:
    #     key =  random.randint(pow(10, 20), q)
    # return key
    x = random.randint(1,p-2) #private key
    y = pow(g, x, p) # public
    return x,y

def encrypt_elgamal(m, y):
    c1 = pow(g, r, p)
    c2 = (m * pow(y, r, p)) % p
    return c1, c2

def decrypt_elgamal(c1, c2, x):
    return (c2 * pow(c1, -1*x, p)) % p

def mul_elgamal(m1_enc, m2_enc):
  return m1_enc[0]*m2_enc[0] % p , m1_enc[1]*m2_enc[1] % p

q = random.randint(pow(10, 20), pow(10, 50))
g = random.randint(2, q) # generator
p = 2083 # prime
r = random.randint(1, p-1)
x,y = keygen_elgamal()