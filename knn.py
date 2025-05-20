import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def KNN(x,y,k):
    dist = [] 
    #Computing Euclidean distance
    dist_ind = np.sqrt(np.sum((x.values-y.values)**2, axis=1))
    #Concatinating the label with the distance
    main_arr = np.column_stack((y_train.values,dist_ind.values))
    #Sorting the distance in ascending order
    main = main_arr[main_arr[:,1].argsort()] 
    #Calculating the frequency of the labels based on value of K
    count = Counter(main[0:k,0])
    keys, vals = list(count.keys()), list(count.values())
    if len(vals)>1:
        if vals[0]>vals[1]:
            return int(keys[0])
        else:
            return int(keys[1])
    else:
        return int(keys[0])


#Read the csv in the form of a dataframe
df= pd.read_csv("data.csv")
df.head()

#Removing the null values
df.dropna(axis=0, inplace=True)
#Reset the index to avoid error
df.reset_index(drop=True, inplace=True)
y = df['RAIN'].replace([False,True],[0,1])
#Removing Date feature and Rain because it is our label
df.drop(['RAIN','DATE'],axis=1,inplace=True) 

#Splitting the data to train(75%) and test(25%)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.25)

pred = KNN(x_test, x_train, 5)


print(classification_report(pred,y_train))