from django.db import models
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os.path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from .takepicture import *
def printmessage():
    print("Reading model.py from machine learning")

# Create your models here.
def decisiontreeclassier():
    dataframe = pd.read_csv('maritalstatus.csv')
    print(dataframe.head())
    print(dataframe.describe())
    print('Dataframe info')
    print(dataframe.info)
    # Data Preprocessing

    # 1. Remove null values from the table
    # print('\nChallenge 6 : Report missing data\n')
    # print(df.isna().sum())
    print("Report missing data in the table :\n")
    print(dataframe.isna().sum())
    dataframe['Gender'].fillna(dataframe['Gender'].mode()[0], inplace=True)
    dataframe['Dependents'].fillna(dataframe['Dependents'].mode()[0], inplace=True)
    dataframe['Married'].fillna(dataframe['Married'].mode()[0], inplace=True)
    dataframe['Self_Employed'].fillna(dataframe['Self_Employed'].mode()[0], inplace=True)
    dataframe['LoanAmount'].fillna(dataframe['LoanAmount'].mode()[0], inplace=True)
    dataframe['Loan_Amount_Term'].fillna(dataframe['Loan_Amount_Term'].mode()[0], inplace=True)
    dataframe['Credit_History'].fillna(dataframe['Credit_History'].mode()[0], inplace=True)
    dataframe['ApplicantIncome'].fillna(dataframe['ApplicantIncome'].mode()[0], inplace=True)
    dataframe['CoapplicantIncome'].fillna(dataframe['CoapplicantIncome'].mode()[0], inplace=True)
    print("Report missing data in the table :\n")
    print(dataframe.isna().sum())

    # Data Discretization
    # Loan_Amount, Applicant_Income,CoApplicantIncome,Loan_Amount_Term
    dataframe['ApplicantIncome'] = pd.cut(dataframe['ApplicantIncome'], bins=[0, 2877, 3812, 5795, 81000],
                                          labels=[1, 2, 3, 4])
    dataframe['CoapplicantIncome'] = pd.cut(dataframe['CoapplicantIncome'], bins=[-1, 1188, 2297, 41667],
                                            labels=[1, 2, 3])
    # dataframe['CoapplicantIncome']=pd.cut(dataframe['CoapplicantIncome'],3)
    dataframe['LoanAmount'] = pd.cut(dataframe['LoanAmount'], bins=[0, 100, 128, 168, 700], labels=[1, 2, 3, 4])
    dataframe['Loan_Amount_Term'] = pd.cut(dataframe['Loan_Amount_Term'], bins=[0, 360, 480], labels=[1, 2])
    print('dataframe applicant income after discretization \n')
    print(dataframe['ApplicantIncome'].describe())
    print(dataframe['ApplicantIncome'].head(50))
    print('dataframe coapplicantIncome after discretizattion \n')
    print(dataframe['CoapplicantIncome'].describe())
    print(dataframe['CoapplicantIncome'].head(50))

    # 3. Data Transformation
    # Data that need to be transform
    # a. Gender , 0= Male, 1= Female
    # b. Married, 0= No, 1= Yes
    # c. Education, 0= Not Graduate, 1= Graduate
    # //print(dataframe['Education'].describe())
    # d. Self_Employed, 0= No , 1= Yes
    # e. Property_Area, 0= Urban, 1= Rural,2 = SemiUrban
    # //print(dataframe['Property_Area'].describe())
    # f. Loan_Status, 0=N , 1 = Y
    print('Married dataset/n/n')
    print(dataframe['Married'])
    Gender_mapping = {'Male': 1, 'Female': 2}
    Married_mapping = {'No': 1, 'Yes': 2}
    Education_mapping = {'Graduate': 1, 'Not Graduate': 2}
    Self_Employed_mapping = {'No': 1, 'Yes': 2}
    Property_Area_mapping = {'Urban': 1, 'Rural': 2, 'SemiUrban': 3}
    Loan_Status_mapping = {'N': 1, 'Y': 2}
    ApplicantIncome_mapping = {'1': 1, '2': 2, '3': 3}

    dataframe['Gender'] = dataframe['Gender'].map(Gender_mapping)
    dataframe['Married'] = dataframe['Married'].map(Married_mapping)
    dataframe['Education'] = dataframe['Education'].map(Education_mapping)
    dataframe['Self_Employed'] = dataframe['Self_Employed'].map(Self_Employed_mapping)
    dataframe['Property_Area'] = dataframe['Property_Area'].map(Property_Area_mapping)
    dataframe['Loan_Status'] = dataframe['Loan_Status'].map(Loan_Status_mapping)
    # dataframe['ApplicantIncome']= dataframe['ApplicantIncome'].map(ApplicantIncome_mapping)
    # dataframe['CoapplicantIncome'] = dataframe['CoapplicantIncome'].map(ApplicantIncome_mapping)
    # dataframe['LoanAmount'] = dataframe['LoanAmount'].map(ApplicantIncome_mapping)
    # dataframe['Loan_Amount_Term']=dataframe['Loan_Amount_Term'].map(ApplicantIncome_mapping)

    print('\nDataframe after data Transformation \n')
    print(dataframe.head())
    print(dataframe.info())

    dataframe_feature = ["Gender", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                         "Loan_Amount_Term", "Credit_History", "Loan_Status"]

    y = dataframe['Married']
    x = dataframe[dataframe_feature]
    print('X : \n\n')
    print(x)
    print('Y : \n\n')
    print(y)

    # Splitting the data into training set and testing data set
    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)
    print('train_x: ')
    print(train_x)
    print(train_x.describe())
    print('Training x null value : ', train_x.isna().sum())
    print('val_x: ')
    print(val_x)
    print('Validate X null value : ', val_x.isna().sum())
    print('train_y : ')
    print(train_y.isna().sum())
    print('val_y : ')
    print(val_y)

    # Training and Validating
    # 1. Training dataset
    print('Dataframe before Training\n')
    print(dataframe.describe())
    decisiontree = DecisionTreeClassifier()
    decisiontree_model = decisiontree.fit(train_x, train_y)
    print('Training Sucessfully')
    # 2. Validating dataset
    print('Validating the model')
    val_predict = decisiontree_model.predict(val_x)
    print('Model validated')

    # Testing accuracy using mean absolute error
    mae = mean_absolute_error(val_y, val_predict)
    print('mean absolute error value : ', mae)

    # Testing accuracy using metrics
    print('Accuracy using metrics', (metrics.accuracy_score(val_y, val_predict) * 100))
    return mae


def GoogleDriveApi():
    tp = takepicture()
    tp.tookpicture()
    print("taking the picture")
