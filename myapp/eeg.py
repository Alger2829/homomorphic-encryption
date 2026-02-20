import random
eeg_g = 7 # generator
eeg_p = 2083 # prime

def is_prime_eeg(n):
    for i in range(2, n): # [2, n-1]
        if n % i == 0:
            return False
    return True


eeg_x = 17 #private key
eeg_y = pow(eeg_g, eeg_x, eeg_p) # public

def encrypt_eeg(m, r ):
    c1 = pow(eeg_g, r, eeg_p)
    c2 = (pow(eeg_g, m, eeg_p) * pow(eeg_y, r, eeg_p)) % eeg_p
    return c1, c2

def decrypt(c1, c2):
    return (c2 * pow(c1, -1*eeg_x, eeg_p)) % eeg_p


# m1 = 9
# r1 = random.randint(1, eeg_p-1)
# m1_encrypted = encrypt_eeg(m1, r1)

# m2 = 11
# r2 = random.randint(1, eeg_p-1)
# m2_encrypted = encrypt_eeg(m2, r2, exponential = True)