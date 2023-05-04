import pandas as pd
from matplotlib import pyplot
from seaborn import lineplot , distplot ,scatterplot , boxplot
dataset=pd.read_csv('Heart_Disease_Prediction2.csv')




#lineplot(data=dataset)

distplot(a=dataset['Sex'] ,hist='True', bins=4)

scatterplot(data=dataset['Bp'])

boxplot(data=dataset['Bp'])
pyplot.show()