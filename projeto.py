#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:45:38 2018

@author: suannyfabyne
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

def correlation_matrix(df):
    corr=df.corr()
    sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    
def plot_box_plot(dataframe_slice, column_name='age', exceptions=[]):
    #gives individual boxplot with all step names
    list_ = dataframe_slice.columns.tolist()
    for column in list_:
        if column in exceptions:
            continue
        else:
            bp = dataframe_slice.boxplot(column=column)
            plt.show(bp)
            plt.clf()
            plt.cla()
            plt.close()
           
def RegBay(Modelo, x_train, y_train, x_test, y_test): 
    model = MultiOutputRegressor(Modelo())
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test) 
    plt.plot(y_test, y_pred, '.')
    xg = np.linspace(0, 10, 1)
    yg = xg
    plt.plot(xg, yg)
    plt.show()
    print(model.score(x_test, y_test))
    print(mean_absolute_error(y_test, y_pred,multioutput='uniform_average'), " Mean Absolute Error")

def RegMPL(Modelo, x_train, y_train, x_test, y_test): 
    model = MultiOutputRegressor(Modelo(random_state=100))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test) 
    plt.plot(y_test, y_pred, '.')
    xg = np.linspace(0, 10, 1)
    yg = xg
    plt.plot(xg, yg)
    plt.show()
    print(model.score(x_test, y_test))
    print(mean_absolute_error(y_test, y_pred,multioutput='uniform_average'), " Mean Absolute Error")
    
def Class(modelo, x_train, y_train, x_test, y_test):
    knn = modelo()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred, average='weighted')) 
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

def CutDataset(dataset, entrada, saida, fora): 
    y = dataset.drop(labels=[entrada[0]], axis=1)
    x = dataset.drop(labels=[saida[0]], axis=1)
    
    for i in entrada[1:]:
        y = y.drop(labels=[i], axis=1)

    for i in saida[1:]:
        x = x.drop(labels=[i], axis=1)  
        
    for i in fora:
        x = x.drop(labels=[i], axis=1)  
        y = y.drop(labels=[i], axis=1)

    return x, y        
    
def NewColumn(corners, data):                 
    for i in range(0, 20):
        corners[i][1] = (corners[i][1]/(len(data)/10))
    data['MeanCornersHome'] = 0
    data['MeanCornersAway'] = 0
    for i in range(0, 20):
        indexH = data.index[data['HomeTeam'] == corners[i][0]].tolist()
        data = data.set_value(indexH, 'MeanCornersHome', corners[i][1], takeable=False)
        indexA = data.index[data['AwayTeam'] == corners[i][0]].tolist()
        data = data.set_value(indexA, 'MeanCornersAway', corners[i][1], takeable=False)
        
#Lendo arquivos
data_0910 = pd.read_csv('09-10.csv')
data_1011 = pd.read_csv('10-11.csv')
data_1112 = pd.read_csv('11-12.csv')
data_1213 = pd.read_csv('12-13.csv')
data_1314 = pd.read_csv('13-14.csv')
data_1415 = pd.read_csv('14-15.csv')
data_1516 = pd.read_csv('15-16.csv')
data_1617 = pd.read_csv('16-17.csv')
data_1718 = pd.read_csv('17-18.csv')
data_1819 = pd.read_csv('18-19.csv')

#Atribuindo coluna Season
data_0910['Season'] = 0
data_1011['Season'] = 1
data_1112['Season'] = 2
data_1213['Season'] = 3
data_1314['Season'] = 4
data_1415['Season'] = 5
data_1516['Season'] = 6
data_1617['Season'] = 7
data_1718['Season'] = 8
data_1819['Season'] = 9

corners1819 =   [['Man City', 77],
                ['Chelsea'	, 56],
                ['Man United',52],
                ['Wolves',52],
                ['Bournemouth',51],
                ['Watford',51],
                ['Everton',49],
                ['Newcastle',48],
                ['Southampton',46],
                ['Tottenham',46],
                ['Liverpool',43],
                ['Crystal Palace',42],
                ['West Ham',42],
                ['Cardiff',40],
                ['Arsenal',36],
                ['Huddersfield',36],
                ['Fulham',35],
                ['Brighton',33],
                ['Leicester',33],
                ['Burnley',25]]

corners1718 =   [['Man City'	, 284],
                ['Tottenham',	246],
                ['Chelsea',	230],
                ['Liverpool',	230],
                ['Southampton',	227],
                ['Arsenal',	225],
                ['Man United',	220],
                ['Bournemouth',	218],
                ['Crystal Palace',	210],
                ['Leicester',	203],
                ['Watford',	183],
                ['West Brom',176],
                ['Burnley',	167],
                ['Newcastle',	167],
                ['Huddersfield',	165],
                ['Brighton'	,163],
                ['West Ham',	161],
                ['Everton',	150],
                ['Swansea',	150],
                ['Stoke',	136]]

corners1617 =  [['Manchester City',	280],
                ['Tottenham Hotspur',	273],
                ['Liverpool',	249],
                ['Arsenal',	227],
                ['Chelsea',	218],
                ['Manchester United',	217],
                ['Crystal Palace',	203],
                ['Southampton',	198],
                ['Leicester City',	197],
                ['Everton',	196],
                ['Swansea City',	196],
                ['AFC Bournemouth',	193],
                ['Stoke City',	188],
                ['Hull City',	179],
                ['West Ham United',	172],
                ['Watford',	164],
                ['Sunderland',	159],
                ['West Bromwich Albion',	159],
                ['Burnley',	149],
                ['Middlesbrough',	141]]

corners1516 =   [['Liverpool',	265],
                ['Manchester City',	257],
                ['Tottenham Hotspur',	254],
                ['Chelsea',	240],
                ['Manchester United',	228],
                ['Arsenal',	227],
                ['West Ham United',	224],
                ['AFC Bournemouth',	221],
                ['Southampton',	220],
                ['Crystal Palace', 219],
                ['Everton',	218],
                ['Leicester City',	197],
                ['Norwich City',	188],
                ['West Bromwich Albion',	188],
                ['Aston Villa',	167],
                ['Watford',	164],
                ['Swansea City',	163],
                ['Newcastle United',	161],
                ['Stoke City',	153],
                ['Sunderland	',153]]

NewColumn(corners1819, data_1819)   
NewColumn(corners1718, data_1718)   
NewColumn(corners1617, data_1617)   
NewColumn(corners1516, data_1516)   

#Atribuindo pesos, repetindo instâncias, para jogos mais recentes valerem mais
for i in range(1,6):
    data_1819=data_1819.append(data_1819) 
 
for i in range(1,3):
    data_1718=data_1718.append(data_1718)

for i in range(1,2):
    data_1617=data_1617.append(data_1617)
    data_1516=data_1516.append(data_1516)
    data_1415=data_1415.append(data_1415)

#Juntando arquivos .csv em uma só tabela
data_0910=data_0910.append(data_1011) 
data_0910=data_0910.append(data_1112) 
data_0910=data_0910.append(data_1213) 
data_0910=data_0910.append(data_1314) 
data_0910=data_0910.append(data_1415) 
data_0910=data_0910.append(data_1516) 
data_0910=data_0910.append(data_1617) 
data_0910=data_0910.append(data_1718) 
dataset=data_0910.append(data_1819) 

#Análise de colunas
dataset = dataset.drop(labels=['Div'], axis=1)
oi = dataset.drop(labels=['Date'], axis=1)
dataset = dataset.drop(labels=['Date'], axis=1)

#Carategorização de colunas
dataset['FTR'] = pd.Categorical(dataset['FTR']).codes
dataset['HTR'] = pd.Categorical(dataset['HTR']).codes
dataset['HomeTeam'] = pd.Categorical(dataset['HomeTeam']).codes
dataset['AwayTeam'] = pd.Categorical(dataset['AwayTeam']).codes
dataset['Referee'] = pd.Categorical(dataset['Referee']).codes


#Renomeando colunas
dataset = dataset.rename(columns={'FTHG': 'FTHomeGoals', 'FTAG': 'FTAwayGoals',
                                  'FTR': 'FTResult', 'HTHG': 'HTHomeGoals', 
                                  'HTAG': 'HTAwayGoals', 'HTR': 'HTResult',
                                  'HS': 'HomeShots', 'AS': 'AwayShots',
                                  'HST': 'HomeTeamShotsTarget', 'AST':'AwayTeamShotsTarget',
                                  'HF': 'HomeTeamFouls', 'AF': 'AwayTeamFouls', 
                                  'HC': 'HomeTeamCorners', 'AC':'AwayTeamCorners',
                                  'HY': 'HomeTeamYellowCards', 'AY': 'AwayTeamYellowCards',
                                  'HR': 'HomeTeamRedCards', 'AR': 'AwayTeamRedCards'})

#Matriz de correlação
correlation_matrix(dataset)

#Outliers
plot_box_plot(dataset)

#Checando atributos com valores incompletos
print(dataset.isnull().sum())

#Removendo linhas duplicadas
duplicated = dataset.duplicated(keep='first').sum()
print(duplicated, "Instâncias duplicadas")

#Removendo saídas que não iremos utilizar
dataset = dataset.drop(labels=['HomeTeamRedCards'], axis=1)
dataset = dataset.drop(labels=['AwayTeamRedCards'], axis=1)
dataset = dataset.drop(labels=['HomeTeamShotsTarget'], axis=1)
dataset = dataset.drop(labels=['AwayTeamShotsTarget'], axis=1)

#PREVENDO GOLS E VENCEDORES

entrada = ['HomeTeam', 'AwayTeam', 'HTResult', 'Referee', 'Season', 'HTHomeGoals', 'HTAwayGoals', 'B365H', 'B365D', 'B365A']
saida = ['FTHomeGoals', 'FTAwayGoals', 'FTResult']
fora = ['HomeShots', 'AwayShots',
        'HomeTeamFouls','AwayTeamFouls', 'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellowCards', 'AwayTeamYellowCards']
x, y = CutDataset(dataset, entrada, saida, fora)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("------------BAYESIANRIDGE-------------")
RegBay(BayesianRidge, x_train, y_train, x_test, y_test)
print("-------DECTREEREG-------")
RegMPL(DecisionTreeRegressor, x_train, y_train, x_test, y_test)
print("--------MLPREGRESSOR-----------")
RegMPL(MLPRegressor, x_train, y_train, x_test, y_test)


entrada = ['HomeTeam', 'AwayTeam', 'Referee', 'Season',  'B365H', 'B365D', 'B365A', 'HTResult','HTHomeGoals', 'HTAwayGoals']
saida = ['FTResult']
fora = ['FTHomeGoals', 'FTAwayGoals','HomeShots', 'AwayShots',
        'HomeTeamFouls','AwayTeamFouls', 'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellowCards', 'AwayTeamYellowCards']
x, y = CutDataset(dataset, entrada, saida, fora)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print("---------DECTREECLASS----------")
Class(DecisionTreeClassifier, x_train, y_train, x_test, y_test)

###################ESCANTEIOS############

entrada = ['HomeTeam', 'AwayTeam', 'HTResult', 'Referee', 'Season', 'HTHomeGoals', 'HTAwayGoals', 'B365H', 'B365D', 'B365A']
saida = ['HomeTeamCorners', 'AwayTeamCorners']
fora = ['FTHomeGoals', 'FTAwayGoals', 'FTResult', 'HomeShots', 'AwayShots',
        'HomeTeamFouls','AwayTeamFouls', 'HomeTeamYellowCards', 'AwayTeamYellowCards']
x, y = CutDataset(dataset, entrada, saida, fora)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print("------------BAYESIANRIDGE-------------")
RegBay(BayesianRidge, x_train, y_train, x_test, y_test)
print("-------DECTREEREG-------")
RegMPL(DecisionTreeRegressor, x_train, y_train, x_test, y_test)
print("--------MLPREGRESSOR-----------")
RegMPL(MLPRegressor, x_train, y_train, x_test, y_test)