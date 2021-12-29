from django.shortcuts import render
import json
import pandas as pd 

# Create your views here.
def home(request):
    # Load the dataset
    df = pd.read_csv("./telecom_users.csv")
    # Grab 10 records
    df = df[:10]
    json_records = df.reset_index().to_json(orient = 'records')
    arr = []
    arr = json.loads(json_records)
    context = {'d': arr}
    return  render(request,'index.html',context)