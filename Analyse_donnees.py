# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:00:46 2017

@author: May-line
"""

import pandas as pd

#train=pd.read_csv('train_users_2.csv')
#test=pd.read_csv('test_users.csv')

#train.values[:,15].value_counts()
histo=pd.value_counts(train.values[:,15])
print(histo)
histo[1:].plot.bar()


import numpy as np

def lecture_fichier():
    
    x_train = []
    y_train = []
    train = pd.read_csv(open("C:/Users/Juliette/Desktop/Projet Kaggle/AirBnb/train_users_2.csv",'r'),parse_dates=[1])
    x_test = pd.read_csv(open("C:/Users/Juliette/Desktop/Projet Kaggle/AirBnb/test_users.csv",'r'),parse_dates=[1])
    x_train = train.iloc[:,:15]
    y_train = train.iloc[:,15]
    
    print(train.iloc[0,:])  
    
    return x_train,y_train,x_test
    
    
def pretraitement(x_train,y_train,x_test):
    
      #suppression colonnes id et date_first_booking
    x_train.drop('id',axis=1,inplace=True)
    x_train.drop('date_first_booking',axis=1,inplace=True)
    
    x_test.drop('id',axis=1,inplace=True)
    x_test.drop('date_first_booking',axis=1,inplace=True)
      

    ### attribut language ###
    
#    x_train[x_train.language!='en'].language.value_counts(dropna=False).plot(kind='bar', color='blue', rot=0) 
#    plt.xlabel('Langue')
#    plt.ylabel("Nombre d'observations")
#    plt.show()
    
    
    #rassemblement de toutes les autres langues que l'anglais dans une catégorie 'autre'
    x_train.loc[x_train.language != 'en','language'] = 'autre'
    x_test.loc[x_test.language != 'en','language'] = 'autre'
#    print(set(x_train[x_train.language!='en'].language))
    
    
    ### destination ###
    
#    destination_pourcentage = y_train[y_train!='NDF'].value_counts(dropna=False)/ y_train[y_train!='NDF'].shape[0] * 100
#    destination_pourcentage.plot(kind='bar',color='#FD5C64',rot=0)
#    plt.xlabel('Pays de destination')
#    plt.ylabel('Pourcentage')
#    plt.show()
    
    
    ### date_account_created ###
    
    #création colonnes year_account_created et month_account_created à partir de date_account_created
    x_train['year_account_created'], x_train['month_account_created'], x_train['day_account_created'] = x_train['date_account_created'].dt.year, x_train['date_account_created'].dt.month, x_train['date_account_created'].dt.day
    x_test['year_account_created'], x_test['month_account_created'], x_test['day_account_created'] = x_test['date_account_created'].dt.year, x_test['date_account_created'].dt.month, x_test['date_account_created'].dt.day
    #suppression colonne date_account_created
    x_train.drop('date_account_created',axis=1,inplace=True)
    x_test.drop('date_account_created',axis=1,inplace=True)
    
    ### timestamp_first_active ###
    x_train['timestamp_first_active'] = pd.to_datetime(x_train['timestamp_first_active'], format='%Y%m%d%H%M%S')
    x_test['timestamp_first_active'] = pd.to_datetime(x_test['timestamp_first_active'], format='%Y%m%d%H%M%S')
    x_train['year_first_active'], x_test['year_first_active'] = x_train['timestamp_first_active'].dt.year, x_test['timestamp_first_active'].dt.year
    x_train['month_first_active'], x_test['month_first_active'] = x_train['timestamp_first_active'].dt.month, x_test['timestamp_first_active'].dt.month
    x_train['day_first_active'], x_test['day_first_active'] = x_train['timestamp_first_active'].dt.day, x_test['timestamp_first_active'].dt.day
    x_train['hour_first_active'], x_test['hour_first_active'] = x_train['timestamp_first_active'].dt.hour, x_test['timestamp_first_active'].dt.hour
    x_train['minute_first_active'], x_test['minute_first_active'] = x_train['timestamp_first_active'].dt.minute, x_test['timestamp_first_active'].dt.minute
    x_train['second_first_active'], x_test['second_first_active'] = x_train['timestamp_first_active'].dt.second, x_test['timestamp_first_active'].dt.second
    
    x_train.drop('timestamp_first_active',axis=1,inplace=True)
    x_test.drop('timestamp_first_active',axis=1,inplace=True)    
    
    ### gender ###
    # regroupement -unknonw- et OTHER dans une classe NaN
    x_train.loc[(x_train.gender != 'MALE') & (x_train.gender !='FEMALE'),'gender'] = np.nan
    x_test.loc[(x_test.gender != 'MALE') & (x_test.gender !='FEMALE'),'gender'] = np.nan
#    print(pd.isnull(np.nan))
#    print(set(x_train.gender))
    
    
    ### age ###
    # tous les âges <15 ou >90 passent à NaN
    x_train.loc[(x_train.age<15) | (x_train.age>90),'age'] = np.nan
#    print(x_train[(x_train.age.notnull())&(x_train.age<15)&(x_train.age>90)])
    #discrétisation en 7 modalités: 15 à 30 ans (noté 15), 30 à 40 (30), 40 à 50 (40), ... , 80 à 90 (80)
    x_train.loc[(x_train.age>=15) & (x_train.age<30),'age'] = 15
    x_train.loc[(x_train.age>=30) & (x_train.age<40),'age'] = 30
    x_train.loc[(x_train.age>=40) & (x_train.age<50),'age'] = 40
    x_train.loc[(x_train.age>=50) & (x_train.age<60),'age'] = 50
    x_train.loc[(x_train.age>=60) & (x_train.age<70),'age'] = 60
    x_train.loc[(x_train.age>=70) & (x_train.age<80),'age'] = 70
    x_train.loc[(x_train.age>=80) & (x_train.age<=90),'age'] = 80
    
    x_test.loc[(x_test.age<15) | (x_test.age>90),'age'] = np.nan
    x_test.loc[(x_test.age>=15) & (x_test.age<30),'age'] = 15
    x_test.loc[(x_test.age>=30) & (x_test.age<40),'age'] = 30
    x_test.loc[(x_test.age>=40) & (x_test.age<50),'age'] = 40
    x_test.loc[(x_test.age>=50) & (x_test.age<60),'age'] = 50
    x_test.loc[(x_test.age>=60) & (x_test.age<70),'age'] = 60
    x_test.loc[(x_test.age>=70) & (x_test.age<80),'age'] = 70
    x_test.loc[(x_test.age>=80) & (x_test.age<=90),'age'] = 80

#    print(y_train)
    print('')     
    print(x_train.iloc[0,:])
    print(y_train.iloc[0])
    print('')
    print(x_test.iloc[0,:])  
    print(x_train.columns==x_test.columns)
    
    return x_train,y_train,x_test
    
    
def main():
    
    x_train,y_train,x_test = lecture_fichier()
    x_train,y_train,x_test = pretraitement(x_train,y_train,x_test)
    

if __name__ == '__main__':
    main()
