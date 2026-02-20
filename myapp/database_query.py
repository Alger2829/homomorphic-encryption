import tenseal as ts


def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

def encrypt_salary(context, salary):
    print("Encrypting salary:", salary)
    encrypted = ts.ckks_vector(context, [salary]).serialize()
    print("Encrypted salary:", encrypted)
    return encrypted

def decrypt_salary(context, encrypted_salary):
    salary = ts.ckks_vector_from(context, encrypted_salary).decrypt()[0]
    print("Decrypted salary:", salary)
    return round(salary,2)

def add_encrypted_values(context, encrypted_salary, value_to_add):
    encrypted_value_to_add = ts.ckks_vector(context, [value_to_add]).serialize()
    encrypted_salary_vector = ts.ckks_vector_from(context, encrypted_salary)
    encrypted_value_vector = ts.ckks_vector_from(context, encrypted_value_to_add)
    result_vector = encrypted_salary_vector + encrypted_value_vector
    return result_vector.serialize()

def multiply_encrypted_values(context, encrypted_salary, value_to_multiply):
    encrypted_value_to_multiply = ts.ckks_vector(context, [value_to_multiply]).serialize()
    encrypted_salary_vector = ts.ckks_vector_from(context, encrypted_salary)
    encrypted_value_vector = ts.ckks_vector_from(context, encrypted_value_to_multiply)
    result_vector = encrypted_salary_vector * encrypted_value_vector
    return result_vector.serialize()

from django import forms
from .models import Employee

class UpdateSalaryForm(forms.Form):
    employee = forms.ModelChoiceField(queryset=Employee.objects.all())
    increment = forms.DecimalField(max_digits=5, decimal_places=2)
    bonus = forms.DecimalField(max_digits=10, decimal_places=2)
