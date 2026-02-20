import json
from django.shortcuts import render
from django.http import JsonResponse

def base(request):
    return render(request, 'base.html')

def index(request):
    return render(request, 'index.html')

def rsa(request):
    return render(request, 'rsa.html')

def pailier(request):
    return render(request, 'pailier.html')

def elgamal(request):
    return render(request, 'elgamal.html')

def eeg(request):
    return render(request, 'eeg.html')

# def fhe(request):
#     return render(request, 'fhe.html')

def comparison(request):
    return render(request, 'comparison.html')

def face(request):
    return render(request, 'face_recognition.html')

def logr(request):
    return render(request, 'logistic_regression.html')

def conv(request):
    return render(request, 'encrypted_convolution.html')

def linr(request):
    return render(request, 'linear_regression.html')

def database(request):
    return render(request, 'database.html')

from django.http import JsonResponse
from .rsa import *
from .pailier import *
from .elgamal import *
from .facereco import *
from .eeg import *
# from .fhe import *

import tenseal as ts
import math
import os

import cv2
import numpy as np
from deepface import DeepFace
from django.conf import settings

#if GPU is interfering uncomment the folowing line
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces TensorFlow to use CPU only

# Initialize encryption context and generate secret and public keys
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

# Serialize and save secret key
secret_context = context.serialize(save_secret_key=True)

# Make context public and serialize
context.make_context_public()
public_context = context.serialize()

def process_uploaded_image(uploaded_image, username):
    print(f"Received username: {username}")

    # Dynamically generate img2_path based on the username
    # img_filename = f"{username}.jpg"
    # img2_path = os.path.join(settings.BASE_DIR, f'myapp/static/myapp/image_database/{img_filename}')

    # # Check if the image file exists for the given username
    # if not os.path.exists(img2_path):
        # return {'success': False, 'message': f'No image found for username {username}', 'img': 'error'}
    img_filename = f'{username}.jpg'
    img2_path = os.path.join(settings.BASE_DIR, f'myapp/static/myapp/image_database/{img_filename}')

    if not os.path.exists(img2_path):
        return {'success': False, 'message': f'No image found for username {username}', 'img': 'error'}

    # Load the uploaded image
    img1 = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    # Load the test image
    img2 = cv2.imread(img2_path)

    try:
        # Extract facial embeddings using DeepFace for both images
        img1_embedding = DeepFace.represent(img1, model_name="Facenet")
        img2_embedding = DeepFace.represent(img2, model_name="Facenet")

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

        # Determine if the images represent the same person
        if euclidean_dist < 10:
            print('The images represent the same person.')
            result = {'success': True, 'message': 'Face Recognised. Authentication Successful.', 'img': 'success'}
        else:
            print('The images do not represent the same person.')
            result = {'success': True, 'message': 'Sorry! We couldn\'t recognize you. Authentication Unsuccessful', 'img': 'unsuccess'}
    except Exception as e:
        result = {'success': False, 'message': str(e), 'img': 'error'}

    return result



def process_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        username = request.POST.get('input1')  # Get the username
        # Image processing code goes here
        result = process_uploaded_image(uploaded_image,username)
        
        return JsonResponse(result)
    else:
        return JsonResponse({'success': False, 'message': 'Invalid request'})
    
#RSA fns
def process_rsa(request):
    if request.method == 'POST':
        num1 = int(request.POST.get('input1'))  # Collect input 1
        num2 = int(request.POST.get('input2'))  # Collect input 2

        
        # Generate keys
        private_key, public_key = keygen_rsa()
        # Multiply and decrypt
        encrypted_num1 = encrypt_rsa(num1, public_key)
        encrypted_num2 = encrypt_rsa(num2, public_key)
        encrypted_product = encrypted_num1*encrypted_num2
        result = decrypt_product_rsa(encrypted_product, private_key)

        # Prepare the response data
        response_data = {
            'encrypted_number1': encrypted_num1,  
            'encrypted_number2': encrypted_num2, 
            'encrypted_product': encrypted_product,
            'private_key': private_key,
            'public_key': public_key,
            'result': result
            
        }

        return JsonResponse(response_data)
    
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})



#Pailier functions
def process_pailier(request):
    if request.method == 'POST':
        num1 = int(request.POST.get('input1'))  # Collect input 1
        num2 = int(request.POST.get('input2'))  # Collect input 2

        # Generate keys
        private_key, public_key = sk,pk
        # Multiply and decrypt
        encrypted_num1 = encrypt_pailier(num1)
        encrypted_num2 = encrypt_pailier(num2)
        encrypted_sum = add__pailier(encrypted_num1,encrypted_num2,public_key)
        encrypted_sprod = scalarmul__pailier(encrypted_num1, num2, public_key)
        decrypted_sum = decrypt__pailier(encrypted_sum, private_key)
        decrypted_sprod = decrypt__pailier(encrypted_sprod,private_key)

        # Prepare the response data
        response_data = {
            'encrypted_number1': encrypted_num1,  
            'encrypted_number2': encrypted_num2, 
            'encrypted_sum': encrypted_sum,
            'encrypted_sprod': encrypted_sprod,
            'private_key': private_key,
            'public_key': public_key,
            'decrypted_sum': decrypted_sum,
            'decrypted_sprod': decrypted_sprod
        }

        return JsonResponse(response_data)
    
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})
    
#El Gamal fns
def process_elgamal(request):
    if request.method == 'POST':
        num1 = int(request.POST.get('input1'))  # Collect input 1
        num2 = int(request.POST.get('input2'))  # Collect input 2

        # Generate keys
        private_key, public_key = x,y
        # Multiply and decrypt
        encrypted_num1 = encrypt_elgamal(num1, public_key)
        encrypted_num2 = encrypt_elgamal(num2, public_key)
        encrypted_c1, encrypted_c2 = mul_elgamal(encrypted_num1, encrypted_num2)
        encrypted_product=(encrypted_c1, encrypted_c2)
        result = decrypt_elgamal(encrypted_c1, encrypted_c2, private_key)

        # Prepare the response data
        response_data = {
            'encrypted_number1': encrypted_num1,  
            'encrypted_number2': encrypted_num2, 
            'encrypted_product': encrypted_product,
            'private_key': private_key,
            'public_key': public_key,
            'result': result
        }

        return JsonResponse(response_data)
    
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})
    

def process_eeg(request):
    if request.method == 'POST':
        num1 = int(request.POST.get('input1'))  # Collect input 1
        num2 = int(request.POST.get('input2'))  # Collect input 2

        r1 = random.randint(1, eeg_p-1)
        r2 = random.randint(1, eeg_p-1)
        # Generate keys
        private_key, public_key = x,y
        # Multiply and decrypt
        encrypted_num1 = encrypt_eeg(num1, r1)
        encrypted_num2 = encrypt_eeg(num2, r2)
        encrypted_sum =  encrypted_num1[0]*encrypted_num2[0] % p , encrypted_num1[1]*encrypted_num2[1] % p
        decrypted_sum = num1+ num2

        # Prepare the response data
        response_data = {
            'encrypted_number1': encrypted_num1,  
            'encrypted_number2': encrypted_num2, 
            'encrypted_sum': encrypted_sum,
            'private_key': private_key,
            'public_key': public_key,
            'decrypted_sum' : decrypted_sum
        }
        return JsonResponse(response_data)
    
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})
    

from myapp.fhe import (
    encrypt_fhe,keygen_fhe,
    decrypt_fhe, add_plain_fhe, mul_plain_fhe,
    add_cipher_fhe, mul_cipher_fhe
)

def fhe(request):
    return render(request, 'fhe.html')

def process_fhe(request):
    if request.method == 'POST':

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

        num1 = int(request.POST.get('input1'))  # Collect input 1
        num2 = int(request.POST.get('input2'))  # Collect input 2
        

        enc_num1 = encrypt_fhe(pk, num1)
        enc_num2 = encrypt_fhe(pk, num2)

        enc_sum = add_cipher_fhe(enc_num1, enc_num2)
        enc_prod = mul_cipher_fhe(enc_num1, num2)

        scalar_sum = add_plain_fhe(enc_num1, num2)
        scalar_prod = mul_plain_fhe(enc_num1, num2)

        dec_sum = decrypt_fhe(sk, enc_sum)
        dec_prod = decrypt_fhe(sk, enc_prod)

        dec_sc_sum = decrypt_fhe(sk, scalar_sum)
        dec_sc_prod = decrypt_fhe(sk, scalar_prod)

        # Prepare the response data
        response_data = {
            'enc_num1': f"({enc_num1[0]}, {enc_num1[1]})", 
            'enc_num2': f"({enc_num2[0]}, {enc_num2[1]})", 
            'enc_sum': f"({enc_sum[0]}, {enc_sum[1]})",
            'enc_prod': f"({enc_prod[0]}, {enc_prod[1]})",
            'scalar_sum': f"({scalar_sum[0]}, {scalar_sum[1]})",
            'scalar_prod': f"({scalar_prod[0]}, {scalar_prod[1]})",
            'decrypted_sum1': dec_sum, 
            'decrypted_prod1': dec_prod, 
            'decrypted_sum2': dec_sc_sum, 
            'decrypted_prod2': dec_sc_prod, 
            'public_key': f"({pk[0]}, {pk[1]})",
            'private_key': f"({sk[0]}, {sk[1]})"
        }
        return JsonResponse(response_data)

    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})

from django.http import JsonResponse

# Initialize TenSEAL context globally
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# Global variables to store votes and candidates
encrypted_votes = {}
candidates = []

def voting(request):
    return render(request, 'voting.html')

def start_voting(request):
    global candidates, encrypted_votes
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            num_voters_str = data.get('numVoters')

            if num_voters_str is None:
                return JsonResponse({'status': 'error', 'message': 'Number of voters is missing'})

            total_voters = int(num_voters_str)
            if total_voters <= 0:
                return JsonResponse({'status': 'error', 'message': 'Invalid number of voters'})

            candidates = ["Siya", "Alger", "Akash"]  # You can make this dynamic if needed

            # Initialize encrypted votes for each candidate
            encrypted_votes.clear()
            encrypted_votes.update({candidate: [ts.ckks_vector(context, [0])] * total_voters for candidate in candidates})

            return JsonResponse({'status': 'voting_started', 'candidates': candidates})

        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON data'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

import base64
import hashlib
def submit_vote(request):
    global encrypted_votes, candidates
    if request.method == 'POST':
        data = json.loads(request.body)  # Parse JSON data
        voter_index = int(data.get('voter_index'))
        vote = int(data.get('vote')) - 1  # Convert to integer and adjust for zero-based indexing

        # Validate the vote
        if 0 <= vote < len(candidates):
            # Encrypt the vote
            for i, candidate in enumerate(candidates):
                if i == vote:
                    encrypted_votes[candidate][voter_index] = ts.ckks_vector(context, [1])  # Vote for the selected candidate
                else:
                    encrypted_votes[candidate][voter_index] = ts.ckks_vector(context, [0])  # Vote for other candidates as 0

            # Encode the encrypted vote using base64
            encrypted_vote = base64.b64encode(encrypted_votes[candidates[vote]][voter_index].serialize()).decode('utf-8')
            
            # Hash the encoded encrypted vote for display
            hashed_encrypted_vote = hashlib.sha256(encrypted_vote.encode()).hexdigest()

            # Return the JsonResponse with both the normal and encrypted votes
            return JsonResponse({'status': 'vote_submitted', 'voted_for': candidates[vote], 'encrypted_vote': hashed_encrypted_vote})
        else:
            return JsonResponse({'status': 'invalid_vote'})
    return JsonResponse({'status': 'error'})


def display_results(request):
    global encrypted_votes, candidates, candidate_counts

    if request.method == 'POST':
        # Count votes
        candidate_counts = {candidate: ts.ckks_vector(context, [0]) for candidate in candidates}
        for candidate, votes in encrypted_votes.items():
            candidate_sum = ts.ckks_vector(context, [0])
            for vote in votes:
                candidate_sum += vote
            candidate_counts[candidate] = candidate_sum

        # Convert candidate counts to encrypted format
        encrypted_counts = {}
        for candidate, count in candidate_counts.items():
            # Serialize the count, encode in base64, and then hash the encoded string
            serialized_count = count.serialize()
            base64_encoded_count = base64.b64encode(serialized_count).decode('utf-8')
            hashed_count = hashlib.sha256(base64_encoded_count.encode()).hexdigest()
            encrypted_counts[candidate] = hashed_count

        return JsonResponse({'status': 'counts_calculated', 'encrypted_counts': encrypted_counts})

    return JsonResponse({'status': 'error'})


def decrypt_results(request):
    global candidates, candidate_counts

    if request.method == 'POST':
        # Initialize decrypted_count dictionary
        decrypted_count = {}

        # Loop through candidate counts
        for candidate, count in candidate_counts.items():
            # Decrypt the CKKS vector directly
            decrypted_vector = count.decrypt()
            # Round the decrypted values and store them in the dictionary
            decrypted_count[candidate] = [round(value) for value in decrypted_vector]

        # Determine the winner
        max_votes = max(sum(count) for count in decrypted_count.values())
        winners = [candidate for candidate, count in decrypted_count.items() if sum(count) == max_votes]

        return JsonResponse({'status': 'decrypted', 'decrypted_counts': decrypted_count, 'winners': winners})

    return JsonResponse({'status': 'error'})

# linear regression model
import pickle
from django.views.decorators.csrf import csrf_exempt

def loadModel(filename):
    # Load the saved model from disk using pickle
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def makePrediction(model, input_data):
    # Perform prediction using the loaded model
    prediction = model.predict([input_data])
    return prediction

def process_linear(request):

    if request.method == 'POST':
        age = int(request.POST.get('age'))  # Collect input 1
        gender = int(request.POST.get('gender')) # Collect input 2
        public_skills = int(request.POST.get('public_skills'))
        tech_skills = int(request.POST.get('tech_skills'))

         # Sample input data for prediction
        input_data = [age, public_skills, tech_skills, gender]  # input data

        # Load the saved model
        loaded_model = loadModel('myapp/data/linear_regression_model.pkl')


        # Make prediction
        predicted_salary = makePrediction(loaded_model, input_data)
    
        # Prepare the response data
        response_data = {
            'predicted_salary': predicted_salary[0]
        }
        print(predicted_salary[0])
        return JsonResponse(response_data)
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})

#logistic regression model
import torch
from sklearn.preprocessing import StandardScaler
from .models import EncryptedLR,LRModel  # Import the EncryptedLR model

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model_state = pickle.load(f)

    model = EncryptedLR(LRModel(9)) #9 == n_features
    model.weight = model_state['weight']
    model.bias = model_state['bias']
    model._delta_w = model_state['_delta_w']
    model._delta_b = model_state['_delta_b']
    model._count = model_state['_count']

    # Load context if available
    ctx_loaded = None
    if model_state['context']:
        ctx_loaded = ts.context_from(model_state['context'])

    return model, ctx_loaded

def process_logistic(request):
    if request.method == 'POST':
        #input tenure, noOfProducts, hasCrCard, balance

        tenure = int(request.POST.get('tenure'))
        noOfProducts = int(request.POST.get('noOfProducts'))
        hasCrCard = request.POST.get('yesno')
        hasCrCard = 1 if hasCrCard.lower() == 'yes' else 0
        balance = float(request.POST.get('balance'))
        creditScore = int(request.POST.get('creditScore'))
        gender = int(request.POST.get('gender'))
        age = int(request.POST.get('age'))
        isActive = int(request.POST.get('isActive'))
        estimSal = float(request.POST.get('estimSal'))
        #static values
        # creditScore = 645
        # gender = 1
        # age =44
        # isActive  =0
        # estimSal = 149756.71

        # Usage
        eelr_loaded, ctx_loaded  = load_model('myapp/data/encrypted_lr_model.pkl')

        # Create array of input
        user_input = np.array([[creditScore, gender, age, tenure, balance, noOfProducts, hasCrCard, isActive, estimSal]])

        # Load the saved scaler
        with open('myapp/data/scaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        user_input_normalized = sc.transform(user_input)

        # Convert normalized data to list (if needed for encryption)
        user_input_normalized_list = user_input_normalized.tolist()[0]

        # Encrypt and predict using the encrypted model
        enc_x_test = ts.ckks_vector(ctx_loaded, user_input_normalized_list)

        enc_prediction = eelr_loaded.predict(enc_x_test)

        # If you need to decrypt the result (assuming it is encrypted)
        prediction = enc_prediction.decrypt()

        # Prepare the response data
        if round(abs(prediction[0])):
            response_data = {
            'exited': 1
            }
        else:
            response_data = {
            'exited': 0
            }
        return JsonResponse(response_data)
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})


#encrypted query
from .database_query import *
context = create_context()
def create_database(request):
    # Create initial database entries with encrypted salaries
    Employee.objects.all().delete()
    if not Employee.objects.exists():
        Employee.objects.bulk_create([
            Employee(name="Eric", age=25, gender='M', encrypted_salary=encrypt_salary(context, 25000)),
            Employee(name="Haley", age=24, gender='F', encrypted_salary=encrypt_salary(context, 28000)),
            Employee(name="Mike", age=26, gender='M', encrypted_salary=encrypt_salary(context, 30000))
        ])
        print('Database created')
    else:
        print('Database already exists')
    return JsonResponse({'status': 'Database created'})

def view_enc_database(request):
    employees = Employee.objects.all().values('name', 'age', 'gender', 'encrypted_salary')
    encrypted_employees = []
    for employee in employees:  
        base64_encoded_salary = base64.b64encode(employee['encrypted_salary']).decode('utf-8')
        hashed_salary = hashlib.sha256(base64_encoded_salary.encode()).hexdigest()
        print(hashed_salary)
        encrypted_employee = {
            'name': employee['name'],
            'age': employee['age'],
            'gender': employee['gender'],
            'encrypted_salary': hashed_salary
        }        
        encrypted_employees.append(encrypted_employee)
    return JsonResponse(encrypted_employees, safe=False)

def view_database(request):
    employees = Employee.objects.all().values('name', 'age', 'gender', 'encrypted_salary')
    decrypted_employees = []
    for employee in employees:
        decrypted_employee = {
            'name': employee['name'],
            'age': employee['age'],
            'gender': employee['gender'],
            'salary': decrypt_salary(context, employee['encrypted_salary'])
        }
        decrypted_employees.append(decrypted_employee)
    return JsonResponse(decrypted_employees, safe=False)

def update_database(request):
    if request.method == 'POST':
        form = UpdateSalaryForm(request.POST)
        if form.is_valid():
            employee = form.cleaned_data['employee']
            increment = float(form.cleaned_data['increment'])
            bonus = float(form.cleaned_data['bonus'])
            # increment = ts.plain_tensor(increment)
            # bonus = ts.plain_tensor(bonus)
            # #this is wrong updation
            # old_salary = ts.ckks_vector_from(context, employee.encrypted_salary)
            # temp = old_salary * increment
            # new_salary = old_salary + temp + bonus
             # Calculate the new salary
            encrypted_increment = multiply_encrypted_values(context, employee.encrypted_salary, 1 + (increment / 100))
            encrypted_new_salary = add_encrypted_values(context, encrypted_increment, bonus)
            employee.encrypted_salary = encrypted_new_salary
            employee.save()
            return JsonResponse({'status': 'Employee updated'})
        else:
            print(form.errors)  # Print form errors for debugging
            return JsonResponse({'status': 'Invalid request', 'errors': form.errors}, status=400)
    return JsonResponse({'status': 'Invalid request'}, status=400)

def show_update_form(request):
    form = UpdateSalaryForm()
    employees = Employee.objects.all()
    options = [{'id': emp.id, 'name': emp.name} for emp in employees]
    return JsonResponse({'form': form.as_p(), 'employees': options})

#encrypted convolution
from .models import ConvNet
from .testconv import load_model,get_title,predict
import matplotlib.pyplot as plt
def process_convolution(request):
    if request.method == 'POST':
        # Take input file from frontend
        uploaded_file = request.FILES['image']
        
        if not uploaded_file.name.endswith('.pt'):
            return JsonResponse({'error': 'Invalid file type. Please upload a .pt file.'})

        # Load the .pt file as a torch tensor directly from the uploaded file
        loaded_image = torch.load(uploaded_file)
        loaded_image_array = loaded_image.numpy().squeeze()

        model = ConvNet()

        # Encryption parameters
        bits_scale = 26
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        )
        context.global_scale = pow(2, bits_scale)
        context.generate_galois_keys()
        # Load the saved encrypted model
        enc_model = load_model('myapp/data/enc_convnet_model.pkl', model)
        # Display the original image; a
        plt.figure(figsize=(4, 4))
        plt.imshow(loaded_image_array, cmap='gray')
        plt.axis('off')
        # plt.show()
        plt.savefig('myapp/static/myapp/output_images/output_image.png')  # Save the image to a file
   
        plt.close()

        # Predict the label using the encrypted model
        kernel_shape = model.conv1.kernel_size
        stride = model.conv1.stride[0]
        predicted_label = predict(context, enc_model, loaded_image, kernel_shape, stride)

        # Get the URL of the saved output image
        # output_image_url = "{% static 'myapp/output_images/output_image.png' %}"
        
        # print(f'The given item is: {get_title(predicted_label)}')
        response_data = {
            "result": get_title(predicted_label),
            # "output_image_url": output_image_url
        }
        return JsonResponse(response_data)
    else:
        # Handle invalid requests
        return JsonResponse({'error': 'Invalid request'})
    




