from tkinter import *
fm =Tk()
fm.title ("Naive-Bayes Algoritm")
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
#-------------------------------------------------------------#
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('Heart_Disease_Prediction2.csv') #dataframe
import matplotlib.pyplot as plt


def predict():
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
   # X_test = sc.transform(X_test)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
   # y_pred = classifier.predict(X_test)
    classifier.fit(x, y)
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
    prediction = classifier.predict(new_point)
    Result = Label(text="Result = " + str(prediction))
    Result.pack()
but =Button(text="Calc", command=predict)

but.pack()
fm.mainloop()
#-----------------------------------------------------


from tkinter import *
fm =Tk()
fm.title ("Knn Algoritm")
fm.minsize(600 ,600)




Label1 =Label(text="Enter Age   ")
Label1.pack()

x1 =Entry()



Label2 =Label(text="Enter gender    ")
Label2.pack()

x2 =Entry()
x2.pack()



Label3=Label(text="Enter Chest_pain_type  ")
Label3.pack()

x3 =Entry()
x3.pack()





Label4 =Label(text="Enter EKG   ")
Label4.pack()

x4 =Entry()
x4.pack()




Label5 =Label(text="Enter Max_HR   ")
Label5.pack()

x5 =Entry()
x5.pack()



Label6 =Label(text="Enter Exercise_angina  ")
Label6.pack()

x6=Entry()
x6.pack()



x7 =Label(text="Thallium    ")
x7.pack()

x7 =Entry()
x7.pack()



Label8 =Label(text="Enter ST_depression    ")
Label8.pack()

x8 =Entry()
x8.pack()


Label9 =Label(text="Enter Slope_of_ST  ")
Label9.pack()

x9 =Entry()
x9.pack()

Label10 =Label(text=" Arrhythmia")
Label10.pack()

x10 =Entry()
x10.pack()


Label11 =Label(text="Enter Cholesterol  ")
Label11.pack()

x11 =Entry()
x11.pack()



Label12 =Label(text="Enter Chest_pain_type   ")
Label12.pack()

x12 =Entry()
x12.pack()



Label13 =Label(text="Enter  FBS ")
Label13.pack()

x13 =Entry()
x13.pack()

#-------------------------------------------------------------#
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Heart_Disease_Prediction2.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()
