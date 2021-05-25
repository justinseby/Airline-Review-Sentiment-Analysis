print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

def food_model(data,target):

    #Train and testing split
    diabetes_X_train, diabetes_X_test,diabetes_y_train, diabetes_y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Save the trained model as a pickle string.
    saved_model = pickle.dumps(regr)

    # Load the pickled model
    regr_from_pickle = pickle.loads(saved_model)

    # Use the loaded pickled model to make predictions
    diabetes_y_pred = regr_from_pickle.predict(diabetes_X_test)

    #sum of predicted output
    sum = np.sum(diabetes_y_pred)

    #print("Printing rating of food")

    #average of predicted output
    average = (sum/len(diabetes_y_pred))
    return average

def boarding_model(data,target):

    #Train and testing split
    diabetes_X_train, diabetes_X_test,diabetes_y_train, diabetes_y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    sum = np.sum(diabetes_y_pred)
    #print("\n")
    #print("Printing rating of boarding")
    average = (sum/len(diabetes_y_pred))
    return average

def infrastructure_model(data,target):

    #Train and testing split
    diabetes_X_train, diabetes_X_test,diabetes_y_train, diabetes_y_test = train_test_split(data, target, test_size=0.2, random_state=0)


    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    sum = np.sum(diabetes_y_pred)

    #print("Printing rating of infrastructure")
    average = (sum/len(diabetes_y_pred))
    return average

def organization_model(data,target):

    #Train and testing split
    diabetes_X_train, diabetes_X_test,diabetes_y_train, diabetes_y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    sum = np.sum(diabetes_y_pred)

    #print("Printing rating of organization")
    average = (sum/len(diabetes_y_pred))
    return average

def payment_model(data,target):

    #Train and testing split
    diabetes_X_train, diabetes_X_test,diabetes_y_train, diabetes_y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    sum = np.sum(diabetes_y_pred)

    #print("Printing rating of payment")
    average = (sum/len(diabetes_y_pred))
    return average

def staff_model(data,target):

    #Train and testing split
    diabetes_X_train, diabetes_X_test,diabetes_y_train, diabetes_y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    sum = np.sum(diabetes_y_pred)

    #print("Printing rating of staff")
    average = (sum/len(diabetes_y_pred))
    return average
    
