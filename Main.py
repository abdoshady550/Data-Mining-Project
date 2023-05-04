import pandas as pd
data=pd.read_csv('Heart_Disease_Prediction.csv')

Chest_pain_type=data['Chest pain type']
Bp=data['BP']
FBS = data['FBS over 120']
EKG =data['EKG results']
Max_HR =data['Max HR']
Exercise_angina=data['Exercise angina']
ST_depression=data['ST depression']
Slope_of_ST=data['Slope of ST']
Thallium=data['Thallium']
Heart_Disease=data['Heart Disease']

Age= data['Age']
gender = data['Sex']
dis=data['Cholesterol']


clean_Age =[]
default_Age=Age.mean()

for c in  Age :
     if  ( str(c)=='nan' or str(c)=='0.0'   )      :
         clean_Age.append(default_Age)
     else :
         clean_Age.append(str(c))

print(clean_Age)
#print(Age)


clean_dis =[]
default_dis=dis.mean()
for d in dis :
     if  ( str(d)=='nan' or str(d)=='0.0')      :
         clean_dis.append(default_dis)
     else :
         clean_dis.append(str(d))

#print(clean_dis)


clean_gender =[]
default_gender =1.0
for n in gender :
     if  (str(n)=='nan') :

         clean_gender.append(default_gender)
     else :
         clean_gender.append(str(n))

#print(clean_gender )




clean_data=pd.DataFrame({'Age':clean_Age ,'Dis':clean_dis ,'gender':clean_gender})

#print(clean_data)
#clean_data.to_csv("Cleaned data.csv")

print(clean_data.iloc[:,])

df=pd.DataFrame({'Age':clean_Age ,
                 'Sex': clean_gender ,
                 'Chest pain type' : Chest_pain_type,
                 'Bp' :  Bp,
                 'Cholesterol':clean_dis,
                 'FBS over 120':FBS,
                 'EKG results':EKG,
                 'Max HR':Max_HR ,
                 'Exercise angina':Exercise_angina,
                 'ST depression':ST_depression ,
                 'Slope of ST':Slope_of_ST,
                 'Thallium':Thallium,
                  'Heart Disease':Heart_Disease
                })

df.to_csv("Heart_Disease_Prediction2.csv")


