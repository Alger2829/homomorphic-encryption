import pandas as pd
import numpy as np
import phe as paillier
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Linear model class
class LinModel:
    def __init__(self):
        pass

    def getResults(self, df):
        y = df.salary
        X = df.drop('salary', axis=1)
        feature_names = X.columns.tolist()  # Get the feature names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        reg = LinearRegression().fit(X_train, y_train)
        reg.feature_names = feature_names  # Assign feature names to the model
        y_pred = reg.predict(X_test)
        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        R = r2_score(y_test, y_pred)
        return reg, y_pred, RMSE, R, X_test, y_test

    def getCoef(self, df):
        return self.getResults(df)[0].coef_

    def saveModel(self, model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)  # Save the model using pickle

# Generate and store keys
def storeKeys():
    public_key, private_key = paillier.generate_paillier_keypair()
    keys = {}
    keys['public_key'] = {'n': public_key.n}
    keys['private_key'] = {'p': private_key.p, 'q': private_key.q}
    with open('custkeys.json', 'w') as file:
        json.dump(keys, file)

# Get keys from file
def getKeys():
    with open('custkeys.json', 'r') as file:
        keys = json.load(file)
        pub_key = paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
        priv_key = paillier.PaillierPrivateKey(pub_key, keys['private_key']['p'], keys['private_key']['q'])
        return pub_key, priv_key

# Serialize data
def serializeData(public_key, data):
    encrypted_data_list = [public_key.encrypt(x) for x in data]
    encrypted_data = {}
    encrypted_data['public_key'] = {'n': public_key.n}
    encrypted_data['values'] = [(str(x.ciphertext()), x.exponent) for x in encrypted_data_list]
    serialized = json.dumps(encrypted_data)
    return serialized

# Get data from JSON file
def getData():
    with open('data.json', 'r') as file:
        d = json.load(file)
    data = json.loads(d)
    return data

# Compute data
def computeData(df):
    data = getData()
    mycoef = LinModel().getCoef(df)
    pk = data['public_key']
    pubkey = paillier.PaillierPublicKey(n=int(pk['n']))
    enc_nums_rec = [paillier.EncryptedNumber(pubkey, int(x[0]), int(x[1])) for x in data['values']]
    results = sum([mycoef[i] * enc_nums_rec[i] for i in range(len(mycoef))])
    return results, pubkey

# Serialize result data
def serializeResultData(df):
    results, pubkey = computeData(df)
    encrypted_data = {}
    encrypted_data['pubkey'] = {'n': pubkey.n}
    encrypted_data['values'] = (str(results.ciphertext()), results.exponent)
    serialized = json.dumps(encrypted_data)
    return serialized

# Load answer from JSON file
def loadAnswer():
    with open('answer.json', 'r') as file:
        ans = json.load(file)
    answer = json.loads(ans)
    return answer

# Main function
def main():
    # Load data
    df = pd.read_csv('employee_data.csv')
    print("Data loaded:")
    print(df)

    # Step 1: Generate keys and serialize input data
    storeKeys()
    pub_key, priv_key = getKeys()
    data = [36, 5, 5, 1]  # Sample input
    datafile = serializeData(pub_key, data)
    with open('data.json', 'w') as file:
        json.dump(datafile, file)

    # Step 2: Compute result using the server logic and serialize result data
    resultfile = serializeResultData(df)
    with open('answer.json', 'w') as file:
        json.dump(resultfile, file)

    # Step 3: Decrypt and print the result
    answer_file = loadAnswer()
    answer_key = paillier.PaillierPublicKey(n=int(answer_file['pubkey']['n']))
    answer = paillier.EncryptedNumber(answer_key, int(answer_file['values'][0]), int(answer_file['values'][1]))
    if (answer_key == pub_key):
        decrypted_result = priv_key.decrypt(answer)
        print("Decrypted result (predicted salary):", decrypted_result)
    else:
        print("Error: Public key mismatch. Unable to decrypt.")

    # Calculate and print accuracy metrics
    lin_model = LinModel()
    reg, y_pred, RMSE, R, X_test, y_test = lin_model.getResults(df)
    print(f"RMSE: {RMSE}")
    print(f"R² score: {R}")

    # Compare actual vs predicted values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(results.head())

    # Save the trained model
    lin_model.saveModel(reg, 'linear_regression_model.pkl')

if __name__ == '__main__':
    main()
