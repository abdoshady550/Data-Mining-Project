
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
dataset = pd.read_csv('Heart_Disease_Prediction2.csv')
print(dataset.head())
transactions=[]
for i in range(0, 100):
    transactions.append([str(dataset.values[i,j])
for j in range(0,10)])
from apyori import apriori
rules= apriori(transactions=transactions, min_support=0.003, min_confidence = 0.2, min_lift=3, min_length=2, max_length=2)
results= list(rules)
results
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")