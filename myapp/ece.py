import random
# EC ElGamal functions#

#using secp256k protocol
#P, the prime number that defines the field and at the sametime decides the curve form
Pcurve = 2**256 - 2**32 - 2**9 - 2**8 - 2**7 - 2**6 - 2**4 - 1

#a’ and ‘b’: define the curve y^2 mod p= x^3 + ax + b mod p and is chosen depending on the security requirement
Acurve = 0
Bcurve = 7

#Generator point, G
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
GPoint = (Gx, Gy)

#Number of points in the field [Order of G]
N =  115792089237316195423570985008687907852837564279074904382605163141518161494337

#h is the cofactor
h = 0o1
k = random.getrandbits(256)

#Compute the modular inverse using the Extended Euclidean Algorithm
def modinv_ece(a, n=Pcurve):
    lm, hm = 1, 0
    low, high = a % n, n
    while low > 1:
        ratio = high / low
        nm, new = hm - lm * ratio, high - low * ratio
        lm, low, hm, high = nm, new, lm, low
    return lm % n

#Elliptic curve addition
def ECadd_ece(a, b):
    LamAdd = ((b[1] - a[1]) * modinv_ece(b[0] - a[0], Pcurve)) % Pcurve
    x = (LamAdd * LamAdd - a[0] - b[0]) % Pcurve
    y = (LamAdd * (a[0] - x) - a[1]) % Pcurve
    return (x, y)

#Point doubling for elliptic curves
def ECdouble_ece(a):
    Lam = ((3 * a[0] * a[0] + Acurve) * modinv_ece((2 * a[1]), Pcurve)) % Pcurve
    x = (Lam * Lam - 2 * a[0]) % Pcurve
    y = (Lam * (a[0] - x) - a[1]) % Pcurve
    return (x, y)

#Double and add multiplication for elliptic curves
def EccMultiply_ece(GenPoint, ScalarHex):
    if ScalarHex == 0 or ScalarHex >= N:
        raise Exception("Invalid Scalar/Private Key")

    ScalarBin = bin(ScalarHex)[2:]
    Q = GenPoint

    for i in range(1, len(ScalarBin)):
        current_bit = ScalarBin[i: i+1]
        Q = ECdouble_ece(Q)
        if current_bit == "1":
            Q = ECadd_ece(Q, GenPoint)
    return Q

#Generate private key
privKey = random.getrandbits(256)

#Generate the public key
def gen_pubKey_ece():
    PublicKey = EccMultiply_ece(GPoint, privKey)
    return PublicKey

#Encrypt a message using ElGamal
def encrypt_ece(msg, Public_Key):
    C1 = EccMultiply_ece(GPoint, k)
    C2 = EccMultiply_ece(Public_Key, k)[0] + msg
    return (C1, C2)

#Decrypt a message using ElGamal
def decryp_ece(C1, C2, private_Key):
    solution = C2 - EccMultiply_ece(C1, private_Key)[0]
    return int(solution)
