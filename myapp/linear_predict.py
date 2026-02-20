import pandas as pd
import pickle

def loadModel(filename):
    # Load the saved model from disk using pickle
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def makePrediction(model, input_data):
    # Perform prediction using the loaded model
    prediction = model.predict([input_data])
    return prediction

def main():
    # Load the saved model
    loaded_model = loadModel('myapp/data/linear_regression_model.pkl')


    # Sample input data for prediction
    input_data = [40, 6, 6, 2]  # Example input data

    # Make prediction
    predicted_salary = makePrediction(loaded_model, input_data)
    print("Predicted salary:", predicted_salary[0])

if __name__ == '__main__':
    main()


