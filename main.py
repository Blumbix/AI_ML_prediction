from matplotlib import pyplot as py
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
gamesFrame = gamesFrame[100:5000]

py.hist(gamesFrame['Global_Sales'],bins=100,rwidth=0.6)
py.show()

X=gamesFrame[['Platform','Year','Genre','Publisher']]
Y=gamesFrame[['Global_Sales']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  random_state=142)

from sklearn.svm import SVR
svr_poly = SVR(gamma='auto')
svr_poly.fit(X_train, Y_train.values.reshape(-1,))
pred=svr_poly.predict(X_test)
#print("Wartosc przewidziana:")
#print(pred[:10])

test = Y_test.to_numpy().ravel()
sum = 0

#Dodac blad procentowy
import math
for i in range(0, len(test)):
    res = math.fabs((test[i]-pred[i])/test[i])*100
    sum += res
    print("Wartosc prawidlowa:",test[i],"\tPrzewidziana: {:.2f}".format(pred[i]),"\tBlad: {:.2f}%".format(res))
acc = sum/len(test)
print("Blad: {:.4f}%".format(acc));
