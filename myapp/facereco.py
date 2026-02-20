import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces TensorFlow to use CPU only

import tenseal as ts
from deepface import DeepFace
import math
import time

def generate_contexts():
    # Initialize encryption context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40

    # Serialize and save secret key
    secret_context = context.serialize(save_secret_key=True)

    # Make context public and serialize
    context.make_context_public()
    public_context = context.serialize()

    return secret_context, public_context

def client_model(img1_path, img2_path, secret_context, public_context):
    print("===== Facial Recognition Using Homomorphic Encryption =====")

    # Extract facial embeddings using DeepFace for image 1
    img1_embedding = DeepFace.represent(img1_path, model_name="Facenet")

    # Extract facial embeddings using DeepFace for image 2
    img2_embedding = DeepFace.represent(img2_path, model_name="Facenet")

    # Load secret key context
    context = ts.context_from(secret_context)

    # Encryption for image 1
    img1_embedding_values_flat = [list(face.values())[0] for face in img1_embedding]
    img1_embedding_values_flat = [val for sublist in img1_embedding_values_flat for val in sublist]
    plain_tensor1 = ts.plain_tensor(img1_embedding_values_flat, dtype="float")
    enc_vector1 = ts.ckks_vector(context, plain_tensor1)

    # Encryption for image 2
    img2_embedding_values_flat = [list(face.values())[0] for face in img2_embedding]
    img2_embedding_values_flat = [val for sublist in img2_embedding_values_flat for val in sublist]
    plain_tensor2 = ts.plain_tensor(img2_embedding_values_flat, dtype="float")
    enc_vector2 = ts.ckks_vector(context, plain_tensor2)

    # Compute squared Euclidean distance between the encrypted vectors
    euclidean_squared = enc_vector1 - enc_vector2
    euclidean_squared = euclidean_squared.dot(euclidean_squared)

    # Decrypt and compute Euclidean distance
    euclidean_dist = math.sqrt(euclidean_squared.decrypt()[0])

    # Output result
    if euclidean_dist < 10:
        print("The images represent the same person.")
    else:
        print("The images do not represent the same person.")

    print("===== End of Facial Recognition System =====")

if __name__ == "__main__":
    img1 = "../downloads/IMG1.jpg"
    img2 = "../downloads/alia3.jpg"
    
    # Generate secret and public contexts
    secret_context, public_context = generate_contexts()
    
    # Perform facial recognition
    client_model(img1, img2, secret_context, public_context)