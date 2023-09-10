# CMPE255-Assignment-2

## AutoML - JADBio
[![JADBio Results](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/jadbio-results.PNG)](https://app.jadbio.com/share/6013d6c6-5a07-4a15-97ee-1840be6671b2/best/overview)

## PyCaret - Low Code Data Mining
### Classification
#### Binary Classification
https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/notebooks/Assignment2_Classification.ipynb
I utilized the same dataset on heart disease patients for classification that I used in the AutoML section of this assignment. This model aims to predict whether or not someone has heart disease based on their Age, Gender, Blood Pressure, Glucose level, troponin level, etc. 

The best model for this dataset that PyCaret found was a Decision Tree Classifier, which had a 97.83% accuracy.
#### Multiclass Classification

#### Regression

### Clustering

### Anomaly Detection

### Association Rules Mining
The current version of PyCaret does not support association rules mining. 
I found an example where PyCaret utilizes the Apriori Algorithm the following way. 
```
from pycaret.arules import *
dataset = get_rules(dataset, transaction_id = 'InvoiceNo', item_id = 'Description')
```
However, when I tried to do the same, I got a module not found error. I tried a few versions of PyCaret and could not find the one where this module existed. 
### Time Series Forecasting
