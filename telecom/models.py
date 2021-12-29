from django.db import models
import pandas as pd
from sklearn.model_selection import train_test_split
from django.shortcuts import render, redirect
import os
from django.http import HttpResponse, JsonResponse
import pickle


# Load the dataset
data = pd.read_csv('./telecom_users.csv')
# Target class
y = data['Churn']
data.drop(['No','customerID','Churn'], axis=1, inplace = True)
X = pd.get_dummies(data, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod'], drop_first = True)
    
X = X.drop(['TotalCharges'], axis=1)
y = y.replace({"Yes":1,"No":0})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
m = {}

# Function to train the model
def training(request):
    # Checking if button is pressed or not
    if 'logit' in request.POST:
        # Import LogisticRegression
        from sklearn.linear_model import LogisticRegression
        # Fit the model
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        filename = 'lg.sav'
        # Dump the model
        pickle.dump(clf, open(filename, 'wb')) 
        return render(request, 'index.html')

    # Checking if button is pressed or not
    if 'xgb' in request.POST:
        # Import XBGClassifier
        from xgboost import XGBClassifier
        # Fit the model
        xgb = XGBClassifier(learning_rate = 0.01, n_estimators=1000).fit(X_train, y_train)
        file_name = 'xgb.sav'
        # Dump the model
        pickle.dump(xgb,open(file_name,'wb'))
        return render(request, 'index.html')

# Function to test the model
def testing(request):
    if 'lg' in request.POST:
        filename = 'lg.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(X_test)

        print(y_pred)
        y_pred = pd.DataFrame(y_pred,columns=['output'])
        y_pred.to_csv('lg.csv')

        filename = 'lg.csv'
        response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')  
        response['Content-Length'] = os.path.getsize(filename)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'lg.csv'
        return response

    if 'xg' in request.POST:
        filename = 'xgb.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(X_test)
        
        print(y_pred)
        y_pred = pd.DataFrame(y_pred,columns=['output'])
        y_pred.to_csv('xgb.csv')
        
        f = 'xgb.csv'
        response = HttpResponse(open(f, 'rb').read(), content_type='text/csv')               
        response['Content-Length'] = os.path.getsize(f)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'xgb.csv'
        return response         

# Function to evaluate the model
def eval(request):
    if 'metric' in request.POST:
        from sklearn.metrics import accuracy_score, precision_score,f1_score
        # Load the pickle
        loaded_model = pickle.load(open('lg.sav', 'rb'))
        # Prediction
        y_pred = loaded_model.predict(X_test)
        # Find the scores
        acc = accuracy_score(y_pred, y_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        # Pass to html
        f = {
            'accuracy':acc,
            'f1':f1
        }

        return render(request,'index.html',f)
    
    if 'xg_metric' in request.POST:
        from sklearn.metrics import accuracy_score, precision_score,f1_score
        loaded_model = pickle.load(open('xgb.sav', 'rb'))
        y_pred = loaded_model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, precision_score,f1_score
        acc = accuracy_score(y_pred,y_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        f = {
            'accuracy':acc,
            'f1':f1
        }
        print(acc)
        # print(m['f1'])
        return render(request, 'index.html', f)