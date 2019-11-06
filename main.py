import pandas
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  r2_score

#reading our CSV file into a pandas DataFrame
d = open('videogames.csv')
gamesFrame = pandas.read_csv(d)
gamesFrame = gamesFrame.dropna()

#transforming columns into a lists of numbers, which go back into our DataFrame
platformList = gamesFrame['Platform'].tolist()
labelEncoderPlatform = LabelEncoder()
labelEncoderPlatform.fit(platformList)
labelsPlatform = labelEncoderPlatform.transform(platformList)
gamesFrame['Platform']=pandas.Series(labelsPlatform)

print(platformList[150:155])
print(labelsPlatform[150:155])

genreList = gamesFrame['Genre'].tolist()
labelEncoderGenre = LabelEncoder()
labelEncoderGenre.fit(genreList)
labelsGenre = labelEncoderGenre.transform(genreList)
gamesFrame['Genre']=pandas.Series(labelsGenre)

publisherList = gamesFrame['Publisher'].tolist()
labelEncoderPublisher = LabelEncoder()
labelEncoderPublisher.fit(publisherList)
labelsPublisher = labelEncoderPublisher.transform(publisherList)
gamesFrame['Publisher']=pandas.Series(labelsPublisher)

gamesFrame = gamesFrame.dropna()

# Keep track of the features
headers = list(gamesFrame)

"""
# Store the revenues of games separately in a list
YIndex = headers.index('Global_Sales')
datasetMatrix = gamesFrame.as_matrix()
print(gamesFrame)
print(datasetMatrix)

# Make the train and test splits
datasetTrain = datasetMatrix[0:1500]
datasetTrainWithoutLabels = np.delete(datasetTrain, YIndex, 1)

labels = datasetTrain[:, YIndex]

datasetTest = datasetMatrix[1500:datasetMatrix.shape[0]]
datasetTestWithoutLabels = np.delete(datasetTest, YIndex, 1)

trueLabels = datasetTest[:, YIndex]
print(trueLabels)
regressor = Lasso()
alphas = np.arange(1,50)

steps = [('regressor',regressor)]
pipeline = Pipeline(steps)

parameterGrid = dict(regressor__alpha = alphas)
GridSearchResult = GridSearchCV(pipeline,param_grid=parameterGrid)

GridSearchResult.fit(datasetTrainWithoutLabels,labels)

#print(GridSearchResult.best_params_)

predictions = GridSearchResult.predict(datasetTestWithoutLabels)
print(r2_score(trueLabels,predictions))


"""