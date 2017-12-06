import numpy as np
from sklearn.neural_network import MLPClassifier
from PIL import Image
from sklearn.metrics import accuracy_score
import glob
import random
import sys

training_root = 'data/mnist_png/training/'

clfName = ''
verbose = False

X_train = []
Y_train = []

def parseArgs():
	clfName = 'classifier'
	verbose = False
	if(len(sys.argv) > 1):
		clfName = sys.argv[1]
	if(len(sys.argv) > 2):
		if(sys.argv[2] == 'True' or sys.argv[2] == 'true'):
			verbose = True
	return clfName, verbose

def getImageAsArray(img):
	image = Image.open(img).convert('L')

	arr = np.asarray(image)

	return [item for sublist in np.asarray(image) for item in sublist]
	#return arr

def fillImageArray(X_train, Y_train):
	print('Log: starting to load training data')
	index = 0
	while(index < 10):
		trainPics = glob.glob(training_root + str(index) + '/*.png')
		for pic in trainPics:
			X_train.append(getImageAsArray(pic))
			Y_train.append(index)
		print('Log: finished parsing training data: ' + str(index))
		index += 1

def train(X_train, Y_train):

	clf = MLPClassifier(activation='tanh', alpha=0.01, solver='sgd', hidden_layer_sizes=(300, 300, 100, 30), verbose=verbose, max_iter=200)
#	clf = MLPClassifier(activation='tanh', alpha=0.01, solver='sgd', hidden_layer_sizes=(50, 20), verbose=verbose, max_iter=10)
	print('Log: training neural net')
	x = X_train[0]
	y = Y_train[0]
	x = np.array(x).reshape(1, -1)
	y = np.array(y).reshape(1, -1)
	clf.fit(x, y)
	print('Log: finished training neural net')

	saveClassifier(clf)
	#sys.popen('python3 test.py ' + clfName)

def saveClassifier(clf):
	from sklearn.externals import joblib
	print('now saving neural net: ' + clfName)
	joblib.dump(clf, 'saves/' + clfName + '.pkl')

clfName, verbose = parseArgs()

fillImageArray(X_train, Y_train)

seedTrain = random.uniform(0, 1)

random.Random(seedTrain).shuffle(X_train)
random.Random(seedTrain).shuffle(Y_train)

train(X_train, Y_train)
