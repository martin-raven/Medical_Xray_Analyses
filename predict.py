from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Sequential
import numpy as np
import os
import cv2


labels=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

test_dir="test_images"

base_mobilenet_model = MobileNet(input_shape =  [128, 128, 1], 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(13, activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.load_weights('xray_class_weights_raven.hdf5')
multi_disease_model.summary()
IMG_SIZE = (128, 128)
x=[]
files=os.listdir(test_dir)
for file in files:
	print("image Detected : ",file)
	im_gray = cv2.imread(test_dir+"/"+file, cv2.IMREAD_GRAYSCALE)
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	x.append(cv2.resize(im_bw, IMG_SIZE, interpolation=cv2.INTER_CUBIC))
x=np.asarray(x)
x=np.reshape(x,(2,128,128,1))
pred_Y=multi_disease_model.predict(x,verbose = True)
for i in range(len(pred_Y)):
	pred_Y[i][7]=0
for i in range(len(pred_Y)):
	print(labels[pred_Y[i].tolist().index(max(pred_Y[i].tolist()))])