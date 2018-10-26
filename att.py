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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
    xg = np.linspace(0, 10, 1, endpoint=False)
    yg = xg
    plt.plot(xg, yg)
    plt.show()
    print(model.score(x_test, y_test))
    print(mean_absolute_error(y_test, y_pred,multioutput='uniform_average'), " Mean Absolute Error")

def RegRS(Modelo, x_train, y_train, x_test, y_test): 
    model = MultiOutputRegressor(Modelo(random_state=100))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test) 
    plt.plot(y_test, y_pred, '.')
    xg = np.linspace(0, 10, 1, endpoint=False)
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
    
def NewColumn(corners, data, colunaHome, colunaAway, colunaTipo):                 
    for i in range(0, 20):
        corners[i][colunaTipo] = (corners[i][colunaTipo]/(len(data)/10))
    data[colunaHome] = 0
    data[colunaAway] = 0
    for i in range(0, 20):
        indexH = data.index[data['HomeTeam'] == corners[i][0]].tolist()
        data = data.set_value(indexH, colunaHome, corners[i][colunaTipo], takeable=False)
        indexA = data.index[data['AwayTeam'] == corners[i][0]].tolist()
        data = data.set_value(indexA, colunaAway, corners[i][colunaTipo], takeable=False)
        
def Fduplicated(x_train, y_train):
    
    season9 = x_train.loc[dataset['Season'] == 9]
    season8 = x_train.loc[dataset['Season'] == 8]
    season7 = x_train.loc[dataset['Season'] == 7]
    
    yseason9 = y_train.loc[dataset['Season'] == 9]
    yseason8 = y_train.loc[dataset['Season'] == 8]
    yseason7 = y_train.loc[dataset['Season'] == 7]
        
    for i in range(1,5):
        season9=season9.append(season9) 
    
    for i in range(1,2):
        season8=season8.append(season8) 
    
    season7=season7.append(season7) 
    
    for i in range(1,5):
        yseason9=yseason9.append(yseason9) 
    
    for i in range(1,2):
        yseason8=yseason8.append(yseason8) 
        
    yseason7=yseason7.append(yseason7) 
        
    x_train = x_train.append(season7)
    x_train = x_train.append(season8)
    x_train = x_train.append(season9)
    y_train = y_train.append(yseason7)
    y_train = y_train.append(yseason8)
    y_train = y_train.append(yseason9)
    
    return x_train, y_train

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

#Corners/Yellow Cards/shots/goals/goals Conceded

corners1819 =   [['Man City', 77, 10, 198, 26, 3],
                ['Everton',59, 13, 123, 15, 12],
                ['Chelsea', 56, 7, 163, 20, 7],
                ['Man United',52, 18, 119, 15, 16],
                ['Wolves',52, 16, 120, 9, 8],
                ['Bournemouth',51, 13, 108, 16, 12],
                ['Watford',51, 19, 110, 13, 12],
                ['Newcastle',48, 10, 92, 6, 14],
                ['Crystal Palace',47, 17, 100, 5, 11],
                ['Southampton',46, 20, 124, 6, 14],
                ['Tottenham',46, 12, 126, 16, 7],
                ['Liverpool',43, 10, 126, 16, 3],
                ['Arsenal',42, 15, 116, 22, 22, 11],
                ['West Ham',42, 25, 95, 8, 14],
                ['Cardiff',40, 13, 108, 8, 19],
                ['Leicester',37, 15, 99, 15, 15],
                ['Huddersfield',36, 13, 89, 4, 18],
                ['Fulham',35, 16, 120, 11, 25],
                ['Brighton',33, 20, 74, 10, 13],
                ['Burnley',25, 18, 78, 10, 17]]

corners1718 =   [['Man City', 284, 59, 665, 106, 27],
                ['Tottenham', 246, 50, 623, 74, 36],
                ['Chelsea',	230, 42, 606, 62, 38],
                ['Liverpool',230, 44, 638, 84, 38],
                ['Southampton',	227, 63, 450, 37, 56],
                ['Arsenal',	225, 57, 594, 74, 51],
                ['Man United', 220, 64, 512, 68, 28],
                ['Bournemouth',	218, 55, 465, 45, 61],
                ['Crystal Palace', 210, 72, 476, 45, 55],
                ['Leicester', 203, 52, 423, 56, 60],
                ['Watford',	183, 63, 440, 44, 64],
                ['West Brom',176, 73, 378, 31, 56],
                ['Burnley',	167, 65, 378, 36, 39],
                ['Newcastle', 167, 52, 451, 39, 47],
                ['Huddersfield', 165, 62, 362, 28, 58],
                ['Brighton'	,163, 54, 384, 34, 54],
                ['West Ham', 161, 73, 372, 48, 68],
                ['Everton', 150, 51, 359, 44, 58],
                ['Swansea', 150, 51, 338, 28, 56],
                ['Stoke', 136, 62, 384, 35, 68]]

corners1617 =  	[['Man City', 280, 71, 633, 80, 39],
                ['Tottenham', 273, 62, 669, 86, 26],
                ['Liverpool', 249, 54, 640, 78, 42],
                ['Arsenal', 227, 68, 566, 77, 44],
                ['Chelsea', 218, 72, 580, 85, 33],
                ['Man United', 217, 78, 591, 54, 29],
                ['Crystal Palace', 203, 77, 439, 50, 63],
                ['Southampton', 198, 59, 550, 41, 48],
                ['Leicester', 197, 72, 433, 48, 63],
                ['Everton', 196, 72, 502, 62, 44],
                ['Swansea', 196, 56, 405, 45, 70],
                ['Bournemouth', 193, 52, 452, 55, 67],
                ['Stoke', 188, 70, 425, 41, 56],
                ['Hull', 179, 67, 397, 37, 80],
                ['West Ham', 172, 78, 499, 47, 64],
                ['Watford', 164, 84, 422, 40, 68],
                ['Sunderland', 159, 78, 387, 29, 69],
                ['West Brom', 159, 80, 399, 43, 51],
                ['Burnley', 149, 65, 391, 39, 55],
                ['Middlesbrough', 141, 77, 351, 27, 53]]

corners1516 =   [['Liverpool', 265, 61, 629, 63, 50],
                ['Man City', 257, 61, 615, 71, 41],
                ['Tottenham', 254, 72, 659, 69, 35],
                ['Chelsea', 240, 58, 526, 59, 53],
                ['Man United', 228, 65, 430, 49, 35],
                ['Arsenal', 227, 40, 571, 65, 36],
                ['West Ham', 224, 58, 558, 65, 51],
                ['Bournemouth', 221, 53, 464, 45, 67],
                ['Southampton', 220, 57, 519, 59, 41],
                ['Crystal Palace', 219, 60, 469, 39, 51],
                ['Everton', 218, 44, 491, 59, 55],
                ['Leicester', 197, 48, 522, 68, 36],
                ['Norwich', 188, 61, 418, 39, 67],
                ['West Brom', 188, 65, 388, 34, 48],
                ['Aston Villa', 167, 75, 380, 27, 76],
                ['Watford', 164, 73, 446, 40, 50],
                ['Swansea', 163, 60, 441, 42, 52],
                ['Newcastle', 161, 60, 397, 44, 65],
                ['Stoke', 153, 51, 419, 41, 55],
                ['Sunderland',153, 64, 439, 48, 62]]

corners1415 =   [['Man City', 277, 77, 670, 83, 38],
				['Arsenal', 254, 68, 610, 71, 36],
				['West Ham', 241, 64, 488, 44, 47],
				['Chelsea', 226, 73, 564, 73, 32],
				['Tottenham', 224, 79, 527, 58, 53],
				['Newcastle', 222, 65, 468, 40, 63],
				['Man United', 214, 64, 512, 62, 37],
				['Leicester', 211, 49, 456, 46, 55],
				['Crystal Palace', 209, 63, 441, 47, 51],
				['Southampton', 208, 57, 509, 54, 33],
				['Liverpool', 198, 66, 590, 52, 48],
				['Stoke', 195, 82, 500, 48, 45],
				['Everton', 194, 66, 483, 48, 50],
				['Sunderland', 186, 94, 408, 31, 53],
				['QPR',	179, 75, 534, 42, 73],
				['Aston Villa', 171, 70, 418, 31, 57],
				['Burnley', 171, 64, 430, 28, 53],
				['West Brom', 171, 64, 412, 38, 51],
				['Hull', 169, 73, 428, 33, 51],
				['Swansea', 151, 48, 426, 46, 49]]

corners1314 =	[['Man City', 283, 72, 673, 102, 37],
				['Everton', 251, 55, 561, 61, 39],
				['Chelsea', 248, 57, 692, 71, 27],
				['Tottenham', 230, 66, 588, 55, 51],
				['Swansea', 226, 55, 496, 54, 54],
				['Liverpool', 224, 54, 651, 101, 50],
				['Man United', 216, 66, 526, 64, 43],
				['Arsenal', 210, 53, 523, 68, 41],
				['Southampton', 210, 60, 534, 54, 46],
				['Norwich', 197, 62,467, 28, 62],
				['Cardiff', 196, 49, 418, 32, 74],
				['Fulham', 194, 58, 431, 40, 85],
				['Newcastle', 190, 53, 579, 43, 59],
				['Sunderland', 188, 59, 491, 41, 60],
				['West Brom', 187, 67, 487, 43, 59],
				['West Ham', 185, 62, 422, 40, 51],
				['Crystal Palace', 172, 58, 414, 33, 48],
				['Stoke', 164, 72, 428, 45, 52],
				['Aston Villa', 163, 78, 431, 39, 61],
				['Hull', 160, 53, 427, 38, 53]]


corners1213	=	[['Liverpool', 284, 54, 739, 71, 43],
				['Man City', 269, 62, 660, 66, 34],
				['Tottenham', 262, 55, 681, 66, 46],
				['Arsenal', 260, 42, 597, 72, 37],
				['Everton', 257, 59, 633, 55, 40],
				['Chelsea', 240, 51, 636, 75, 39],
				['Man United', 218, 57, 561, 86, 43],
				['West Ham', 212, 74, 492, 45, 53],
				['Southampton', 209, 43, 516, 49, 60],
				['Newcastle', 203, 71, 533, 45, 68],
				['West Brom', 202, 63, 506, 53, 57],
				['Wigan', 198, 66, 500, 47, 73],
				['Swansea', 193, 58, 504, 47, 51],
				['Reading', 189, 45, 394, 43, 73],
				['Sunderland', 179, 62, 417, 41, 54],
				['Aston Villa', 177, 72, 438, 47, 69],
				['Fulham', 176, 48, 460, 50, 60],
				['Norwich', 172, 60, 431, 41, 58],
				['QPR', 170, 59, 500, 30, 60],
				['Stoke', 165, 78, 390, 34, 45]]


corners1112 =	[['Liverpool', 309, 53, 667, 47, 40],
				['Man United', 279, 51, 646, 89, 33],
				['Tottenham', 279, 43, 701, 66, 41],
				['Man City', 269, 51, 738, 93, 29],
				['Arsenal', 262, 64, 637, 74, 49],
				['Chelsea', 253, 74, 671, 65, 46],
				['Aston Villa', 218, 70, 438, 37, 53],
				['West Brom', 212, 48, 544, 45, 52],
				['Wigan', 212, 67, 519, 42, 62],
				['Bolton', 210, 50, 495, 46, 77],
				['Wolves', 205, 64, 473, 40, 82],
				['Swansea', 201, 40, 472, 44, 51],
				['QPR', 195, 54, 539, 43, 66],
				['Fulham', 187,  54, 541, 48, 51],
				['Sunderland', 181, 60, 458, 45, 46],
				['Everton', 180, 60, 520, 50, 40],
				['Newcastle', 171, 67, 489, 56, 51],
				['Blackburn', 166, 66, 453, 48, 78],
				['Stoke', 166, 60, 376, 36, 53],
				['Norwich', 165, 58, 514, 52, 66]]

corners1011 =	[['Chelsea', 257, 59, 745, 69, 33],
				['Tottenham', 256, 51, 657, 55, 46], 
				['Arsenal', 252, 65, 654, 72, 43],
				['Everton', 245, 55, 580, 51, 45],
				['Man United', 243, 56, 618, 78, 37],
				['Wolves', 241, 62, 459, 46, 66],
				['Aston Villa', 230, 71, 506, 48, 59],
				['Man City', 222, 71, 546, 60, 33],
				['Liverpool', 212, 63, 582, 59, 44],
				['Newcastle', 202, 78, 507, 56, 57],
				['Bolton', 197, 67, 570, 52, 56],
				['Fulham', 191, 52, 547, 49, 43],
				['Stoke', 191, 68, 482, 46, 48],
				['West Brom', 187, 52, 597, 56, 71],
				['Blackpool', 183, 47, 531, 55, 78],
				['Sunderland', 181, 57, 532, 45, 56],
				['West Ham', 181, 59, 572, 43, 70],
				['Blackburn', 173, 65, 453, 46, 59],
				['Wigan', 169, 67, 511, 40, 61],
				['Birmingham', 150, 57, 401, 37, 58]]

corners0910 = 	[['Man United', 297, 49, 695, 86, 28],
				['Chelsea', 286, 54, 834, 103, 32],
				['Liverpool', 268, 55, 642, 61, 35],
				['Arsenal', 257, 56, 660, 83, 41],
				['Tottenham', 241, 58, 681, 67, 41],
				['Man City', 237, 49, 526, 73, 45],
				['Aston Villa', 236, 59, 497, 52, 39],
				['Everton', 230, 57, 589, 60, 49],
				['Wolves', 204, 63, 436, 32, 56],
				['Portsmouth', 189, 68, 533, 34, 66],
				['Birmingham', 186, 74, 452, 38, 47],
				['Bolton', 183, 74, 558, 42, 67],
				['Blackburn', 181, 57, 489, 41, 55],
				['Stoke', 181, 63, 402, 34, 48],
				['Wigan', 181, 65, 555, 37, 79],
				['Sunderland', 180, 77, 444, 48, 56],
				['West Ham', 180, 62, 549, 47, 66],
				['Fulham', 177, 46, 463, 39, 46],
				['Burnley', 174, 57, 459, 42, 82],
				['Hull', 150, 64, 388, 34, 75]]

#ESCANTEIOS
NewColumn(corners1819, data_1819, 'MeanCornersHome', 'MeanCornersAway', 1)   
NewColumn(corners1718, data_1718, 'MeanCornersHome', 'MeanCornersAway', 1)   
NewColumn(corners1617, data_1617, 'MeanCornersHome', 'MeanCornersAway', 1)   
NewColumn(corners1516, data_1516, 'MeanCornersHome', 'MeanCornersAway', 1)   
NewColumn(corners1415, data_1415, 'MeanCornersHome', 'MeanCornersAway', 1)
NewColumn(corners1314, data_1314, 'MeanCornersHome', 'MeanCornersAway', 1)
NewColumn(corners1213, data_1213, 'MeanCornersHome', 'MeanCornersAway', 1)
NewColumn(corners1112, data_1112, 'MeanCornersHome', 'MeanCornersAway', 1)
NewColumn(corners1011, data_1011, 'MeanCornersHome', 'MeanCornersAway', 1)
NewColumn(corners0910, data_0910, 'MeanCornersHome', 'MeanCornersAway', 1)

#CARTOES AMARELOS
NewColumn(corners1819, data_1819, 'MeanCardsHome', 'MeanCardsAway', 2)   
NewColumn(corners1718, data_1718, 'MeanCardsHome', 'MeanCardsAway', 2)   
NewColumn(corners1617, data_1617, 'MeanCardsHome', 'MeanCardsAway', 2)    
NewColumn(corners1516, data_1516, 'MeanCardsHome', 'MeanCardsAway', 2)    
NewColumn(corners1415, data_1415, 'MeanCardsHome', 'MeanCardsAway', 2) 
NewColumn(corners1314, data_1314, 'MeanCardsHome', 'MeanCardsAway', 2) 
NewColumn(corners1213, data_1213, 'MeanCardsHome', 'MeanCardsAway', 2) 
NewColumn(corners1112, data_1112, 'MeanCardsHome', 'MeanCardsAway', 2) 
NewColumn(corners1011, data_1011, 'MeanCardsHome', 'MeanCardsAway', 2) 
NewColumn(corners0910, data_0910, 'MeanCardsHome', 'MeanCardsAway', 2) 

#CHUTES
NewColumn(corners1819, data_1819, 'MeanShotsHome', 'MeanShotsAway', 3)    
NewColumn(corners1718, data_1718, 'MeanShotsHome', 'MeanShotsAway', 3)    
NewColumn(corners1617, data_1617, 'MeanShotsHome', 'MeanShotsAway', 3)    
NewColumn(corners1516, data_1516, 'MeanShotsHome', 'MeanShotsAway', 3)    
NewColumn(corners1415, data_1415, 'MeanShotsHome', 'MeanShotsAway', 3) 
NewColumn(corners1314, data_1314, 'MeanShotsHome', 'MeanShotsAway', 3) 
NewColumn(corners1213, data_1213, 'MeanShotsHome', 'MeanShotsAway', 3) 
NewColumn(corners1112, data_1112, 'MeanShotsHome', 'MeanShotsAway', 3) 
NewColumn(corners1011, data_1011, 'MeanShotsHome', 'MeanShotsAway', 3) 
NewColumn(corners0910, data_0910, 'MeanShotsHome', 'MeanShotsAway', 3) 

#GOLS
NewColumn(corners1819, data_1819, 'MeanGoalsHome', 'MeanGoalsAway', 4)    
NewColumn(corners1718, data_1718, 'MeanGoalsHome', 'MeanGoalsAway', 4)     
NewColumn(corners1617, data_1617, 'MeanGoalsHome', 'MeanGoalsAway', 4)      
NewColumn(corners1516, data_1516, 'MeanGoalsHome', 'MeanGoalsAway', 4)      
NewColumn(corners1415, data_1415, 'MeanGoalsHome', 'MeanGoalsAway', 4)   
NewColumn(corners1314, data_1314, 'MeanGoalsHome', 'MeanGoalsAway', 4)   
NewColumn(corners1213, data_1213, 'MeanGoalsHome', 'MeanGoalsAway', 4)   
NewColumn(corners1112, data_1112, 'MeanGoalsHome', 'MeanGoalsAway', 4)   
NewColumn(corners1011, data_1011, 'MeanGoalsHome', 'MeanGoalsAway', 4)   
NewColumn(corners0910, data_0910, 'MeanGoalsHome', 'MeanGoalsAway', 4)   

#GOLS SOFRIDOS
NewColumn(corners1819, data_1819, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)      
NewColumn(corners1718, data_1718, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)     
NewColumn(corners1617, data_1617, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)     
NewColumn(corners1516, data_1516, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)     
NewColumn(corners1415, data_1415, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)  
NewColumn(corners1314, data_1314, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)  
NewColumn(corners1213, data_1213, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)  
NewColumn(corners1112, data_1112, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)  
NewColumn(corners1011, data_1011, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)  
NewColumn(corners0910, data_0910, 'MeanGoalsConHome', 'MeanGoalsConAway', 5)  

#Atribuindo pesos, repetindo instâncias, para jogos mais recentes valerem mais
#for i in range(1,6):
#    data_1819=data_1819.append(data_1819) 
 
#for i in range(1,3):
#    data_1718=data_1718.append(data_1718)

#for i in range(1,2):
#    data_1617=data_1617.append(data_1617)
#    data_1516=data_1516.append(data_1516)
#    data_1415=data_1415.append(data_1415)

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
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

dataset['FTR'] = label_encoder.fit_transform(dataset['FTR'])
dataset['FTR'] = dataset['FTR'].reshape(len(dataset['FTR']), 1)
onehot_encoded = onehot_encoder.fit_transform(dataset['FTR'])

dataset['HTR'] = label_encoder.fit_transform(dataset['HTR'])
dataset['HTR'] = dataset['HTR'].reshape(len(dataset['HTR']), 1)
onehot_encoded = onehot_encoder.fit_transform(dataset['HTR'])

dataset['HomeTeam'] = label_encoder.fit_transform(dataset['HomeTeam'])
dataset['HomeTeam'] = dataset['HomeTeam'].reshape(len(dataset['HomeTeam']), 1)
onehot_encoded = onehot_encoder.fit_transform(dataset['HomeTeam'])

dataset['AwayTeam'] = label_encoder.fit_transform(dataset['AwayTeam'])
dataset['AwayTeam'] = dataset['AwayTeam'].reshape(len(dataset['AwayTeam']), 1)
onehot_encoded = onehot_encoder.fit_transform(dataset['AwayTeam'])

dataset['Referee'] = label_encoder.fit_transform(dataset['Referee'])
dataset['Referee'] = dataset['Referee'].reshape(len(dataset['Referee']), 1)
onehot_encoded = onehot_encoder.fit_transform(dataset['Referee'])




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
#correlation_matrix(dataset)

#Outliers
#plot_box_plot(dataset)

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


dataset.reset_index(drop=True, inplace=True)


######ESCANTEIOS
entrada = ['HomeTeam', 'AwayTeam', 'Season', 'MeanCornersHome', 'MeanCornersAway', 'MeanShotsHome', 'MeanShotsAway']     
saida = ['HomeTeamCorners', 'AwayTeamCorners']
fora = ['FTHomeGoals', 'FTAwayGoals','HomeShots', 'AwayShots', 'HomeTeamFouls','AwayTeamFouls', 'FTResult', 'HomeTeamYellowCards', 'B365H', 'B365D', 'B365A', 'AwayTeamYellowCards',  'MeanCardsHome', 'MeanCardsAway', 'MeanGoalsHome', 'MeanGoalsAway','MeanGoalsConHome', 'MeanGoalsConAway', 'Referee', 'HTResult','HTHomeGoals','HTAwayGoals']

x, y = CutDataset(dataset, entrada, saida, fora)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x_train, y_train = Fduplicated(x_train, y_train)

RegBay(RandomForestRegressor, x_train, y_train, x_test, y_test)
RegBay(BayesianRidge, x_train, y_train, x_test, y_test)
RegRS(MLPRegressor, x_train, y_train, x_test, y_test)


###### NUM GOLS
entrada = ['HomeTeam', 'AwayTeam', 'Season', 'MeanCornersHome', 'MeanCornersAway', 'MeanShotsHome', 'MeanShotsAway', 'B365H', 'B365D', 'B365A', 'MeanGoalsHome', 'MeanGoalsAway','MeanGoalsConHome', 'MeanGoalsConAway', 'HTResult','HTHomeGoals','HTAwayGoals']     
saida = ['HomeTeamCorners', 'AwayTeamCorners']
fora = ['FTHomeGoals', 'FTAwayGoals','HomeShots', 'AwayShots', 'HomeTeamFouls','AwayTeamFouls', 'FTResult', 'HomeTeamYellowCards',  'AwayTeamYellowCards',  'MeanCardsHome', 'MeanCardsAway',  'Referee']

x, y = CutDataset(dataset, entrada, saida, fora)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x_train, y_train = Fduplicated(x_train, y_train)

RegBay(RandomForestRegressor, x_train, y_train, x_test, y_test)
RegBay(BayesianRidge, x_train, y_train, x_test, y_test)
RegRS(MLPRegressor, x_train, y_train, x_test, y_test)