# CMPE255-Assignment-2

## AutoML - [JADBio Results](https://app.jadbio.com/share/6013d6c6-5a07-4a15-97ee-1840be6671b2/best/overview)
I used JADBios AutoML functionality to create an ML Model that classifies whether or not a patient has heart disease. This model can use a patient's age, gender, bp, glucose level, troponin level, etc. to predict whether or not they are positive  for heart disease.

[![JADBio Results](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/jadbio-results.PNG)](https://app.jadbio.com/share/6013d6c6-5a07-4a15-97ee-1840be6671b2/best/overview)

JADBio found that a Random Forest Classifier was the best-performing model for this dataset. It achieved a 98.5% accuracy. I used this same data set for classification in Pycaret and JADBio achieved better results. The features this model depended on the most were troponin levels, kcm, and age. 
## PyCaret - Low Code Data Mining
### Binary Classification - [Notebook](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/notebooks/Assignment2_Classification.ipynb)
I utilized the same dataset on heart disease patients for classification that I used in the AutoML section of this assignment. This model aims to predict whether or not someone has heart disease based on their Age, Gender, Blood Pressure, Glucose level, troponin level, etc. 

The best model for this dataset that PyCaret found was a Decision Tree Classifier, which had a 97.83% accuracy.
This model found that a patient's troponin levels and KCM were the most important features for identifying a patient with heart disease. 

### Multiclass Classification - 
I utilized a dataset that magnetic mine detection sensor data. Included in this data is the voltage measured, the height of the sensor, the type of soil, and the type of land mine, if one was found. We will use this dataset to classify the type of mine based on sensor data. The possible categories are 1. None, 2. Anti-Tank, 3. Anti-Personel, 4. Booby Trapped Anti-Personnel and 5. M14 Anti-Personnel

The confusion matrix shows that our model is very good at detecting when there is no mine or when there is an anti-tank mine. The model is also good at detecting whether a mine is anit-personnel, but it gets confused when determining the specific type of anti-personnel mine.       

There were few features, and the feature importance plot shows that all are important, however, the sensed voltage was the most influential.

The ROC Curves for XGBClassifier further demonstrate that the model is good at detecting no mine and anti-tank mines. And that it has trouble distinguishing between the different types of anti-personnel mines.

### Regression - [Notebook](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/notebooks/Assignment2_Regression.ipynb)
I utilized a dataset on used cars and the price for which they sold. The aim of this data set is to train a model that can predict the sale price of a used car. The dataset included a vast number of car descriptors. Fuel type, aspiration, number of doors, body type, wheelbase, engine size, engine displacement, engine cylinders, and much more could all have an effect on the sale price of a used vehicle. 

PyCaret found that a Huber Regressor was the best model for predicting sales price given this data. It identified horsepower, car length, engine size, and fuel efficiency to be some of the most influential features for predicting a car's sales price. 

### Clustering - [Notebook](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/notebooks/Assignment2_Clustering.ipynb)
I utilized a dataset of credit card customers for this clustering example. The dataset includes a customer's credit limit, how many cards they have, how many times they have visited a bank online and in person, and how many calls to banks they have made. We will be using this data to cluster these customers with other customers who behave similarly. 

For this model, I decided to use KMeans Clustering. 
- The Elbow plot shows that the ideal number of clusters is 4. 
- The silhouette plot shows how well each sample in each cluster is matched to its cluster. From our silhouette plot, we can see that the majority of samples are well-matched with only a very small amount having negative values. 

Using our Kmeans model we can predict which data points belong in which of the four clusters. New data points could also be fit to our clusters.

### Anomaly Detection - [Notebook](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/notebooks/Assignment2_AnomalyDetection.ipynb)
I utilized a dataset that contains the data on traffic in NYC. This can be used to find on which days traffic in New York is abnormal. This dataset includes a timestamp and a value that represents the number of taxi passengers in New York in a given 30-minute timeframe. 

I tried various models for anomaly detection but ended up utilizing Iforest. The dataset included traffic data for the NYC marathon, Thanksgiving, Christmas, New Year's day, and a Snowstorm. Other people have used this data set with anomaly detection and these dates were expected to be the anomolies. My model, however, seemed to pick out several dates with no relation to these occurrences that would be expected to make a difference in traffic. 

### Association Rules Mining
The current version of PyCaret does not support association rules mining. 
I found an example where PyCaret utilizes the Apriori Algorithm the following way. 
```
from pycaret.arules import *
dataset = get_rules(dataset, transaction_id = 'InvoiceNo', item_id = 'Description')
```
However, when I tried to do the same, I got a module not found error. I tried a few versions of PyCaret and could not find the one where this module existed. 

### Time Series Forecasting - [Notebook](https://github.com/sjkchang/CMPE255-Assignment-2/blob/master/notebooks/Assignment_2_Time_Series.ipynb)
To conduct Time Series Forecasting, I utilized a dataset that included stock price information on Amazons stock starting in 1997 ending in 2020. The goal of this is to predict the closing price of Amazon's stock in the future. 

Pycaret found that the best forecaster for this dataset was a naive forecaster. The prediction only forecasted the stock price for a short while ahead of the final datapoint, so the information is not very useful. 
