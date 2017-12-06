import numpy as np
from sklearn.neural_network import MLPClassifier
from PIL import Image
import sys
	
	
def parseImage(imageStr):
	image = Image.open(imageStr)
	
	image = image.convert("L")
	
	image = image.resize((28, 28), Image.NEAREST)

	image.save('gr.png')
	
	arr = np.asarray(image)

	return [item for sublist in np.asarray(image) for item in sublist]	

def loadClassifier(clfName):
	from sklearn.externals import joblib
	print('loading Classifier: ' + clfName)
	return joblib.load('saves/' + clfName + '.pkl')	

if(len(sys.argv) > 2):
	clfString = sys.argv[1]
	imageStr = sys.argv[2]
	imgArr = parseImage(imageStr)
	clf = loadClassifier(clfString)
	res = clf.predict(imgArr)
	print(res)