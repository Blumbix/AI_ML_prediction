import math
import pandas
from matplotlib import pyplot as py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#wczytanie danych z pliku .csv do pandas DataFrame
d = open('videogames.csv')
gamesFrame = pandas.read_csv(d)
gamesFrame = gamesFrame.dropna()

#Zamiana kolumn tekstowych na numeryczne
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

#Usuniecie pustych wartosci
gamesFrame = gamesFrame.dropna()
#Wybranie przedzialu na ktorym pracujemy
gamesFrame = gamesFrame[500:10000]

#Statystyki naszych danych
print("Srednia:",gamesFrame['Global_Sales'].mean())
sigma = 0.0
for x in gamesFrame['Global_Sales']:
    sigma += (x - gamesFrame['Global_Sales'].mean())**2
w=sigma/len(gamesFrame['Global_Sales'])
print("Wariancja:",w)
print("Odchylenie standardowe:",math.sqrt(w))

#Tworzenie histogramu danych wyjsciowych
py.hist(gamesFrame['Global_Sales'],bins=100,rwidth=0.6)
py.show()

#Glowna czesc - podzielenie na dane wejsciowe i wyjsciowe oraz predykcja z regresja
X=gamesFrame[['Platform','Year','Genre','Publisher']]
Y=gamesFrame[['Global_Sales']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  random_state=40)

from sklearn import neural_network
mlp_reg = neural_network.MLPRegressor(hidden_layer_sizes=(100, 100, 100,), activation='relu', solver='adam',
                                      learning_rate='adaptive', alpha=0.0001, max_iter=200, learning_rate_init=0.001)
mlp_reg.fit(X_train, Y_train.values.reshape(-1,))
pred=mlp_reg.predict(X_test)
#print("Wartosc przewidziana:")
#print(pred[:10])

test = Y_test.to_numpy().ravel()
sum = 0

#Obliczany blad procentowy
for i in range(0, len(test)):
    res = math.fabs((test[i]-pred[i])/test[i])*100
    sum += res
    print("Wartosc prawidlowa:",test[i],"\tPrzewidziana: {:.2f}".format(pred[i]),"\t  Blad: {:.2f}%".format(res))
acc = sum/len(test)
print("Blad: {:.4f}%".format(acc));
