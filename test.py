import numpy as np
from sklearn.neural_network import MLPClassifier
from PIL import Image
from sklearn.metrics import accuracy_score
import glob
import random
import sys

testing_root = 'data/mnist_png/testing/'

X_test = []
Y_test = []

clfName = ''

def parseArgs():
	clfName = 'classifier'
	if(len(sys.argv)):
		clfName = sys.argv[1]
	return clfName

def getImageAsArray(img):
	image = Image.open(img).convert('L')

	arr = np.asarray(image)

	return [item for sublist in np.asarray(image) for item in sublist]

def fillImageArray(X_test, Y_test):
	print('Log: now loading Testing Data')
	index = 0
	while(index < 10):
		testPics = glob.glob(testing_root + str(index) + '/*.png')
		for pic in testPics:
			X_test.append(getImageAsArray(pic))
			Y_test.append(index)
		print('Log: finished parsing testing data: ' + str(index))
		index += 1


def loadClassifier():
	from sklearn.externals import joblib
	print('loading Classifier: ' + clfName)
	return joblib.load('saves/' + clfName + '.pkl')	
	
def testClassifier(clf, X_test, Y_test):
	res = clf.predict(X_test)
	print('Accuracy of Classifier ' + clfName + ': ' + str(accuracy_score(Y_test, res)))

clfName = parseArgs()

fillImageArray(X_test, Y_test)

seedTest = random.uniform(0, 1)

random.Random(seedTest).shuffle(X_test)
random.Random(seedTest).shuffle(Y_test)

clf = loadClassifier()

testClassifier(clf, X_test, Y_test)
