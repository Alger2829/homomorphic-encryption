import math
import random


def is_prime__pailier(n):
    for i in range(2, n): # [2, n)
        if n % i == 0:
            return False
    return True

def generate_prime__pailier(min_value, max_value):
    prime = random.randint(min_value, max_value)
    while not is_prime__pailier(prime):
        prime = random.randint(min_value, max_value)
    return prime

def lx__pailier(x):
    y = (x-1)/n
    assert y - int(y) == 0
    return int(y)

def keygen_pailier():
  p, q = generate_prime__pailier(1000,5000), generate_prime__pailier(1000,5000)
  while p == q:
    q = generate_prime__pailier(1000,5000)
  # p=13
  # q=17
  n = p*q
  phi = (p-1)*(q-1)
  assert math.gcd(n, phi) == 1
  g = n + 1
  lmbda = phi * 1
  mu = pow(lmbda, -1, n)
  return ((n,g) , (lmbda,mu))

def encrypt_pailier(m):
    assert math.gcd(r, n) == 1
    c = ( pow(g, m, n*n) * pow(r, n, n*n) ) % (n*n)
    return c

def decrypt__pailier(c,sk):
    lmbda = sk[0]
    mu=sk[1]
    p = ( lx__pailier(pow(c, lmbda, n*n)) * mu ) % n
    return p

def add__pailier(c1,c2,pk):
  n = pk[0]
  return (c1*c2) % (n*n)

def scalarmul__pailier(c1,m1,pk):
  n = pk[0]
  return pow(c1, m1, n*n)

pk, sk = keygen_pailier()
n=pk[0]
g=pk[1]
r = random.randint(0, n)
while(math.gcd(r, n) != 1):
  r = random.randint(0, n)
