import csv
import numpy as np
import pandas as pd
import math

def loaddata():
    df = pd.read_csv (r'ratings.csv')
    popularity= df.groupby("movieId").count()
    average_ratings= df.groupby("movieId").mean()
    data = {
        #"movieId": popularity['movieId'],
        "popularity": popularity['userId'],
        "average_ratings": average_ratings['rating']}
    df = pd.concat(data,
               axis = 1)
    mv = pd.read_csv (r'movies.csv')
    finalmv= mv.merge(df, how='inner', on='movieId')
    return finalmv

def dataprocessing(finalmv):
    movies=[]
    m,n= finalmv.shape
    for i in range(m):
        movierow=np.zeros(20)
        movierow[0]=finalmv['movieId'][i]
        genres=finalmv['genres'][i].split("|")
        for genre in genres:
            if genre=='Action':
                movierow[1]=1
            elif genre=='Adventure':
                movierow[2]=1
            elif genre=='Animation':
                movierow[3]=1
            elif genre=='Children\'s':
                movierow[4]=1
            elif genre=='Crime':
                movierow[5]=1
            elif genre=='Documentary':
                movierow[6]=1
            elif genre=='Fantasy':
                movierow[7]=1
            elif genre=='Film-Noir':
                movierow[8]=1
            elif genre=='Horror':
                movierow[9]=1
            elif genre=='Musical':
                movierow[10]=1
            elif genre=='Mystery':
                movierow[11]=1
            elif genre=='Romance':
                movierow[12]=1
            elif genre=='Sci-Fi':
                movierow[13]=1
            elif genre=='Thriller':
                movierow[14]=1
            elif genre=='War':
                movierow[15]=1
            elif genre=='Western':
                movierow[16]=1
            else:
                movierow[17]=1
            movierow[18]=finalmv['popularity'][i]
            movierow[19]=finalmv['average_ratings'][i]
        movies.append(movierow)
    mv= np.array(movies)
    trainclassLabel=mv[:,0]
    epsilon=1e-100
    normalize_train= (np.delete(mv,0,axis=1)-np.delete(mv,0,axis=1).mean(axis=0))/(np.delete(mv,0,axis=1).std(axis=0)+epsilon)
    normalize_train= np.column_stack((normalize_train,trainclassLabel))
    return normalize_train

def findtestpoint(movieId, data):
    return data[np.where(np.isin(data[:,-1], [movieId]))][0]

def findmoviename(movieId,df):
    m= df[df['movieId']==movieId]
    name=" ".join(m['title'].values[0].split(" ")[0:len(m['title'].values[0].split(" "))-1 ])
    return name

def findmovieId(title,df):
    id=df[df['title'].str.contains(title, na=False,  case=False)]
    return id['movieId'].values[0]

def eucledianDistance(point1, point2):
    sum=0
    for i in range (len(point1)-1):
        sum= sum + (point1[i]-point2[i])**2
    return math.sqrt(sum)

def getNeighbours(testpoint, traindata, index):
    mn=99999
    lb=-1
    idx=-1
    for i in range (len(traindata)):
        dist=eucledianDistance(testpoint,traindata[i])
        if(mn>dist and (i not in index) and(traindata[i][19]!=testpoint[19])):
            mn=dist
            idx=i
            lb=traindata[i][len(traindata[i])-1]
    return [lb,idx]

def KNN(testdata,traindata,k):
    neighboursLabel=[]
    index=[]
    for j in range (k):
        neighbours= getNeighbours(testdata,traindata,index)
        neighboursLabel.append(int(neighbours[0]))
        index.append(neighbours[1])
    return neighboursLabel

def main():
    data= loaddata()
    name=input("Enter Your Movie Name : ")
    movieId=findmovieId(name,data)
    train_data=dataprocessing(data)
    test_data=findtestpoint(movieId,train_data)
    recommend=KNN(test_data,train_data,10)
    print("Recommended Movies : ")
    for i in range(len(recommend)):
        print(f"{'%3d' %(i+1)}: {findmoviename(recommend[i],data)}")       


main()
