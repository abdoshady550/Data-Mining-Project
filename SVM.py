from tkinter import *
fm =Tk()
fm.title ("SVM Algoritm")
fm.minsize(600 ,600)



Label1 =Label(text="Enter Age   ")
Label1.pack()
n1 =Entry()
n1.pack()


Label2 =Label(text="Enter gender    ")
Label2.pack()
n2 =Entry()
n2.pack()


Label3=Label(text="Enter Chest_pain_type  ")
Label3.pack()
n3 =Entry()
n3.pack()

Label4 =Label(text="Enter EKG   ")
Label4.pack()
n4 =Entry()
n4.pack()


Label5 =Label(text="Enter Max_HR   ")
Label5.pack()
n5 =Entry()
n5.pack()

Label6 =Label(text="Enter Exercise_angina  ")
Label6.pack()
n6=Entry()
n6.pack()



Label7 =Label(text="Thallium    ")
Label7.pack()
n7 =Entry()
n7.pack()



Label8 =Label(text="Enter ST_depression    ")
Label8.pack()
n8 =Entry()
n8.pack()


Label9 =Label(text="Enter Slope_of_ST  ")
Label9.pack()
n9 =Entry()
n9.pack()

Label10 =Label(text=" Arrhythmia")
Label10.pack()
n10 =Entry()
n10.pack()


Label11 =Label(text="Enter Cholesterol  ")
Label11.pack()
n11 =Entry()
n11.pack()



Label12 =Label(text="Enter Chest_pain_type   ")
Label12.pack()

n12 =Entry()
n12.pack()



Label13 =Label(text="Enter  FBS ")
Label13.pack()
n13 =Entry()
n13.pack()

import pandas as pd
import numpy as np
from sklearn import svm

df=pd.read_csv("Heart_Disease_Prediction2.csv") #dataframe

#df.shape
#df.head()

def predict():
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    z = svm.SVC(kernel='linear', C=1.0)

    z.fit(x, y)
    new_x1 =int(n1.get())
    new_x2 = int(n2.get())
    new_x3 = int(n3.get())
    new_x4 = int(n4.get())
    new_x5 = int(n5.get())
    new_x6 = int(n6.get())
    new_x7 = int(n7.get())
    new_x8 = int(n8.get())
    new_x9 = int(n9.get())
    new_x10 = int(n10.get())
    new_x11 = int(n11.get())
    new_x12 = int(n12.get())
    new_x13 = int(n13.get())



    new_point = [(
                  new_x1, new_x2 , new_x3 , new_x4 , new_x5 , new_x6
                  , new_x7 , new_x8  , new_x9 , new_x10 , new_x11 , new_x12 , new_x13 )]
    prediction = z.predict(new_point)
    Result = Label(text="Result = " + str(prediction) )
    Result.pack()

but =Button(text="Calc", command=predict)

but.pack()
fm.mainloop()





#-------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import svm

df=pd.read_csv("Heart_Disease_Prediction2.csv") #dataframe

#df.shape
#df.head()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)


z =svm.SVC(kernel='linear', C=1.0)

z.fit(x_train,y_train)
y_pred=z.predict(x_test)
df=pd.DataFrame({'Actual':y_test , 'Predicted':y_pred})
print(df)

#df.to_csv("SVM.csv")



from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print("accuracy =" , acc)




