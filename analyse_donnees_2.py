
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:44:52 2017

@author: Juliette
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

def lecture_fichier():
    
    x_train = []
    y_train = []
    train = pd.read_csv(open("C:/Users/Juliette/Desktop/Projet Kaggle/AirBnb/train_users_2.csv",'r'),parse_dates=[1])
    
    #### suppression des observations où country_destination = 'NDF'
    print(np.shape(train))    
    train = train[train.country_destination != 'NDF']
    print(np.shape(train)) 
    
    x_test = pd.read_csv(open("C:/Users/Juliette/Desktop/Projet Kaggle/AirBnb/test_users.csv",'r'),parse_dates=[1])
    x_train = train.iloc[:,:15]
    y_train = train.iloc[:,15]
    
    print(train.iloc[0,:])  
    
    return x_train,y_train,x_test
    
    
def graphe_destination(y_train):
    
    destination_pourcentage = y_train[y_train!='NDF'].value_counts(dropna=False)/ y_train[y_train!='NDF'].shape[0] * 100
    destination_pourcentage.plot(kind='bar',color='#FD5C64',rot=0)
    plt.xlabel('Pays de destination')
    plt.ylabel('Pourcentage')
    plt.show()
    
    
def pretraitement(x): #### argument : soit x_train, soit x_test !!
    
      #suppression colonnes id et date_first_booking
    x.drop('id',axis=1,inplace=True)
    x.drop('date_first_booking',axis=1,inplace=True)
    

    ### attribut language ###
    
#    x[x.language!='en'].language.value_counts(dropna=False).plot(kind='bar', color='blue', rot=0) 
#    plt.xlabel('Langue')
#    plt.ylabel("Nombre d'observations")
#    plt.show()
    
    #rassemblement de toutes les autres langues que l'anglais dans une catégorie 'autre'
    x.loc[x.language != 'en','language'] = 'autre'
#    print(set(x[x.language!='en'].language))
    
    
    ### date_account_created ###
    
    #création colonnes year_account_created et month_account_created à partir de date_account_created
    x['year_account_created'], x['month_account_created'], x['day_account_created'] = x['date_account_created'].dt.year, x['date_account_created'].dt.month, x['date_account_created'].dt.day
    #suppression colonne date_account_created
    x.drop('date_account_created',axis=1,inplace=True)
    
    ### timestamp_first_active ###
    x['timestamp_first_active'] = pd.to_datetime(x['timestamp_first_active'], format='%Y%m%d%H%M%S')
    x['year_first_active'] = x['timestamp_first_active'].dt.year
    x['month_first_active'] = x['timestamp_first_active'].dt.month
    x['day_first_active'] = x['timestamp_first_active'].dt.day
    x['hour_first_active'] = x['timestamp_first_active'].dt.hour
    x['minute_first_active'] = x['timestamp_first_active'].dt.minute
    x['second_first_active'] = x['timestamp_first_active'].dt.second
    
    x.drop('timestamp_first_active',axis=1,inplace=True) 
    
    
    ### gender ###
    
    # regroupement -unknown- et OTHER dans une classe NaN
    x.loc[(x.gender != 'MALE') & (x.gender !='FEMALE'),'gender'] = np.nan
#    print(pd.isnull(np.nan))
#    print(set(x.gender))
    
    
    ### age ###
    
    # tous les âges <15 ou >90 passent à NaN
    x.loc[(x.age<15) | (x.age>90),'age'] = np.nan
#    print(x_train[(x_train.age.notnull())&(x_train.age<15)&(x_train.age>90)])
    
    #discrétisation en 7 modalités: 15 à 30 ans (noté 15), 30 à 40 (30), 40 à 50 (40), ... , 80 à 90 (80)
    x.loc[(x.age>=15) & (x.age<30),'age'] = 15
    x.loc[(x.age>=30) & (x.age<40),'age'] = 30
    x.loc[(x.age>=40) & (x.age<50),'age'] = 40
    x.loc[(x.age>=50) & (x.age<60),'age'] = 50
    x.loc[(x.age>=60) & (x.age<70),'age'] = 60
    x.loc[(x.age>=70) & (x.age<80),'age'] = 70
    x.loc[(x.age>=80) & (x.age<=90),'age'] = 80
       
    
    return x
    

def binarisation(x):
    
     
#    cols = x.columns
#    print('cols',cols)
#    print(cols[:3])
#    cols = cols[3]|cols[:3]|cols[4:]
#    print(cols)
#    
#    print('col : ',x.columns)
#    signup_f = x.iloc[:,3].reshape(-1,1)
#    
#    enc = OneHotEncoder()
#    enc.fit(signup_f)
#    
#    print(enc.n_values_)
#    print(enc.get_params())
#    print(signup_f)
#    print(x.iloc[:,3])
    
    
    ##
    
    
#    
#    print(x)
    
    #encodage variables catégorielles avec type de valeurs = chaines de caractères
    attributs = x.transpose().to_dict().values() #transposition
    
    Dvec = DictVectorizer() #Pour binariser quand les valeurs sont des chaines de caractères
    attributs = Dvec.fit_transform(attributs).toarray()
    print(attributs)
    print(Dvec.get_feature_names()) #OK !
    feature_names = Dvec.get_feature_names()
    indice_signup_flow = feature_names.index('signup_flow')
    print('indice : ',indice_signup_flow) #112
    print(np.shape(attributs)) #(213451, 118)
    print(type(attributs))
    print('att : ',attributs)
    
#    enc = OneHotEncoder(categorical_features=[indice_signup_flow],handle_unknown='error', n_values='auto', sparse=True)
#    enc.fit(att)
#    print(enc.n_values_)
#    print(enc.feature_indices_)
#    enc.transform(att).toarray()
#    
#    print(type(att[:,indice_signup_flow]))
#    print(np.shape(att[:,indice_signup_flow]))
#    signup_flow = att[:,indice_signup_flow]
#    print(type(signup_flow))
#    print(np.shape(signup_flow))
#    
#    print(signup_flow)
    
#    print(Dvec.get_feature_names())
#    print(type(att))
#    print(np.shape(att))
    
    #encodage variable catégorielle avec type de valeurs = nombres
    signup_flow = attributs[:,indice_signup_flow].reshape(-1,1)
    enc = OneHotEncoder() #Pour binariser quand les valeurs sont des nombres
    enc.fit_transform(signup_flow)
    print(enc.n_values_)
    print(enc.get_params())
    print(signup_flow[:100])
       
#    print(x.iloc[0,:])

    return x


def binarisation2(x,x_test):
    
     
#    cols = x.columns
#    print('cols',cols)
#    print(cols[:3])
#    cols = cols[3]|cols[:3]|cols[4:]
#    print(cols)
#    
#    print('col : ',x.columns)
#    signup_f = x.iloc[:,3].reshape(-1,1)
#    
#    enc = OneHotEncoder()
#    enc.fit(signup_f)
#    
#    print(enc.n_values_)
#    print(enc.get_params())
#    print(signup_f)
#    print(x.iloc[:,3])
    
    
    ##
    
    
#    
#    print(x)
    
    #encodage variables catégorielles avec type de valeurs = chaines de caractères
    attributs = x.transpose().to_dict().values() #transposition
    att_test = x_test.transpose().to_dict().values()
    
    Dvec = DictVectorizer()
    Dvec.fit(attributs)
    attributs = Dvec.transform(attributs).toarray().tolist()
    att_test = Dvec.transform(att_test).toarray().tolist() #pour avoir le même nb de colonnes en train et test
#    print(attributs)
    attributs = [[int(i) for i in j] for j in attributs] #### PB AVEC LES NA
    
#    print(attributs)
    print(Dvec.get_feature_names()) #OK !
    feature_names = Dvec.get_feature_names()
    indice_signup_flow = feature_names.index('signup_flow')
    print('indice : ',indice_signup_flow) #89
    print(np.shape(attributs)) #(62096,96)
    print(np.shape(att_test))
    print(type(attributs))
    print('attributs : ',attributs[0])
    
#    enc = OneHotEncoder(categorical_features=[indice_signup_flow],handle_unknown='error', n_values='auto', sparse=True)
#    enc.fit(att)
#    print(enc.n_values_)
#    print(enc.feature_indices_)
#    enc.transform(att).toarray()
#    
#    print(type(att[:,indice_signup_flow]))
#    print(np.shape(att[:,indice_signup_flow]))
#    signup_flow = att[:,indice_signup_flow]
#    print(type(signup_flow))
#    print(np.shape(signup_flow))
#    
#    print(signup_flow)
    
#    print(Dvec.get_feature_names())
#    print(type(att))
#    print(np.shape(att))
    
    ##encodage variable catégorielle avec type de valeurs = nombres
    signup_flow = attributs[:,indice_signup_flow].reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit_transform(signup_flow)
    print(enc.n_values_)
    print(enc.get_params())
    print(signup_flow[:100])
       
#    print(x.iloc[0,:])

    return attributs,att_test
    
    
def main():
    
    x_train,y_train,x_test = lecture_fichier()
#    graphe_destination(y_train)
    x_train = pretraitement(x_train)
    x_test = pretraitement(x_test)
#    binarisation(x_train)
#    binarisation(x_test)
    

if __name__ == '__main__':
    main()
            
