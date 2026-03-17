from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

main = tkinter.Tk()
main.title("A Hybrid model for Classification of Liver Tumors Across Sequential and Time-Variant Data to Assist Doctors in Treatment Using SVM and Comparing Error Rate with k-Nearest Neighbors")
main.geometry("1300x1200")

global filename
global classifier
global svm_er, knn_er
global X, Y
global X_train, X_test, y_train, y_test
global pca

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def splitDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    text.delete('1.0', END)
    
    # Assuming features for liver images are stored in 'features' directory
    X = np.load('features/X.txt.npy')  # You need to ensure this path is correct for liver data
    Y = np.load('features/Y.txt.npy')
    X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3])))
    
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    print(X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Total Liver CT/MRI Images Found in dataset: " + str(len(X)) + "\n")
    text.insert(END, "Train split dataset to 80%: " + str(len(X_train)) + "\n")
    text.insert(END, "Test split dataset to 20%: " + str(len(X_test)) + "\n")

def executeSVM():
    global classifier
    global svm_er
    text.delete('1.0', END)
    
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    svm_er = 1 - (accuracy_score(y_test, predict))
    classifier = cls
    text.insert(END, "SVM Error Rate: " + str(svm_er) + "\n")

def executeKNN():
    global knn_er
    cls = KNeighborsClassifier(n_neighbors=2)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    knn_er = 1 - accuracy_score(y_test, predict)
    text.insert(END, "KNN Error Rate: " + str(knn_er) + "\n")

def predictCancer():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(64, 64, 3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr / 255
    test = []
    test.append(im2arr)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0], (test.shape[1] * test.shape[2] * test.shape[3])))
    test = pca.transform(test)
    
    predict = classifier.predict(test)[0]
    msg = ''
    if predict == 0:
        msg = "Uploaded Liver Scan is Normal"
    if predict == 1:
        msg = "Uploaded Liver Scan has Tumor"
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)

def graph():
    height = [svm_er, knn_er]
    bars = ('SVM Error Rate', 'KNN Error Rate')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='A Hybrid model for Classification of Liver Tumors Across Sequential and Time-Variant Data to Assist Doctors in Treatment Using SVM and Comparing Error Rate with k-Nearest Neighbors')
title.config(bg='deep sky blue', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Liver Cancer Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

readButton = Button(main, text="Read & Split Dataset to Train & Test", command=splitDataset)
readButton.place(x=350, y=550)
readButton.config(font=font1)

svmButton = Button(main, text="Execute SVM Algorithm", command=executeSVM)
svmButton.place(x=50, y=600)
svmButton.config(font=font1)

kmeansButton = Button(main, text="Execute KNN Algorithm", command=executeKNN)
kmeansButton.place(x=350, y=600)
kmeansButton.config(font=font1)

predictButton = Button(main, text="Predict Liver Tumor", command=predictCancer)
predictButton.place(x=50, y=650)
predictButton.config(font=font1)

graphButton = Button(main, text="Error Rate Graph", command=graph)
graphButton.place(x=350, y=650)
graphButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential

main = tkinter.Tk()
main.title("Detection of Liver Cancer from CT/MRI Image using SVM Classification and Compare the Survival Rate of Patients using 3D Convolutional Neural Network (3D CNN) on Liver Tumor Dataset")
main.geometry("1300x1200")

global filename
global classifier
global svm_sr, cnn_sr
global X, Y
global X_train, X_test, y_train, y_test

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def splitDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    text.delete('1.0', END)
    
    # Assuming liver features are in the 'features' folder
    X = np.load('features/X.txt.npy')  # Path to liver dataset features
    Y = np.load('features/Y.txt.npy')  # Path to liver dataset labels
    X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3])))
    
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    print(X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Total Liver CT/MRI Images Found in dataset: " + str(len(X)) + "\n")
    text.insert(END, "Train split dataset to 80%: " + str(len(X_train)) + "\n")
    text.insert(END, "Test split dataset to 20%: " + str(len(X_test)) + "\n")

def executeSVM():
    global classifier
    global svm_sr
    text.delete('1.0', END)
    
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    svm_sr = accuracy_score(y_test, predict) * 100
    classifier = cls
    text.insert(END, "SVM Survival Rate: " + str(svm_sr) + "\n")

def executeCNN():
    global cnn_sr
    X = np.load('features/X.txt.npy')  # Path to liver dataset features
    Y = np.load('features/Y.txt.npy')  # Path to liver dataset labels
    Y = to_categorical(Y)  # Convert labels to categorical (one-hot encoding)
    
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dense(2, activation='softmax'))  # Assuming 2 classes: Normal/Abnormal
    
    print(classifier.summary())
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
    hist = hist.history
    acc = hist['accuracy']
    cnn_sr = acc[9] * 100
    text.insert(END, "CNN Survival Rate: " + str(cnn_sr) + "\n")

def predictCancer():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))  # Resize for model input
    im2arr = np.array(img)
    im2arr = im2arr.reshape(64, 64, 3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr / 255  # Normalize the image
    
    test = []
    test.append(im2arr)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0], (test.shape[1] * test.shape[2] * test.shape[3])))
    test = pca.transform(test)  # Apply PCA if needed
    
    predict = classifier.predict(test)[0]
    msg = ''
    if predict == 0:
        msg = "Uploaded Liver Scan is Normal"
    if predict == 1:
        msg = "Uploaded Liver Scan is Abnormal"
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)

def graph():
    height = [svm_sr, cnn_sr]
    bars = ('SVM Survival Rate', 'CNN Survival Rate')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='Detection of Liver Cancer from CT/MRI Image using SVM Classification and Compare the Survival Rate of Patients using 3D Convolutional Neural Network (3D CNN) on Liver Tumor Dataset')
title.config(bg='deep sky blue', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Liver Cancer Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

readButton = Button(main, text="Read & Split Dataset to Train & Test", command=splitDataset)
readButton.place(x=350, y=550)
readButton.config(font=font1)

svmButton = Button(main, text="Execute SVM Algorithm", command=executeSVM)
svmButton.place(x=50, y=600)
svmButton.config(font=font1)

cnnButton = Button(main, text="Execute CNN Algorithm", command=executeCNN)
cnnButton.place(x=350, y=600)
cnnButton.config(font=font1)

predictButton = Button(main, text="Predict Liver Cancer", command=predictCancer)
predictButton.place(x=50, y=650)
predictButton.config(font=font1)

graphButton = Button(main, text="Survival Rate Graph", command=graph)
graphButton.place(x=350, y=650)
graphButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential

main = tkinter.Tk()
main.title("Prediction of Time-to-Event Outcomes in Diagnosing Liver Cancer Based on SVM and Compare the Accuracy of Predicted Outcome with Deep CNN Algorithm")
main.geometry("1300x1200")

global filename
global classifier
global svm_acc, cnn_acc
global X, Y
global X_train, X_test, y_train, y_test
global pca

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def splitDataset():
    global X, Y, X_train, X_test, y_train, y_test, pca
    try:
        X = np.load('features/X.txt.npy')
        Y = np.load('features/Y.txt.npy')
    except FileNotFoundError:
        text.insert(END, "Error: Feature files not found. Generate features first.\n")
        return
    text.delete('1.0', END)
    
    # Update paths to liver cancer dataset
    X = np.load('features/liver_X.txt.npy')  # Path to liver dataset features
    Y = np.load('features/liver_Y.txt.npy')  # Path to liver dataset labels
    
    X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3])))
    
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    print(X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Total CT/MRI Images Found in dataset: " + str(len(X)) + "\n")
    text.insert(END, "Train split dataset to 80%: " + str(len(X_train)) + "\n")
    text.insert(END, "Test split dataset to 20%: " + str(len(X_test)) + "\n")

def executeSVM():
    global classifier
    global svm_acc
    text.delete('1.0', END)
    
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    svm_acc = accuracy_score(y_test, predict) * 100
    classifier = cls
    text.insert(END, "SVM Accuracy: " + str(svm_acc) + "\n")

def executeCNN():
    global cnn_acc
    X = np.load('features/liver_X.txt.npy')  # Path to liver dataset features
    Y = np.load('features/liver_Y.txt.npy')  # Path to liver dataset labels
    Y = to_categorical(Y)  # Convert labels to categorical (one-hot encoding)
    
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dense(2, activation='softmax'))  # Assuming 2 classes: Normal/Abnormal
    
    print(classifier.summary())
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=12, shuffle=True, verbose=2)
    hist = hist.history
    acc = hist['accuracy']
    cnn_acc = acc[9] * 100
    text.insert(END, "CNN Accuracy: " + str(cnn_acc) + "\n")

def predictCancer():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))  # Resize for model input
    im2arr = np.array(img)
    im2arr = im2arr.reshape(64, 64, 3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr / 255  # Normalize the image
    
    test = []
    test.append(im2arr)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0], (test.shape[1] * test.shape[2] * test.shape[3])))
    test = pca.transform(test)  # Apply PCA if needed
    
    predict = classifier.predict(test)[0]
    msg = ''
    if predict == 0:
        msg = "Uploaded Liver Scan is Normal"
    if predict == 1:
        msg = "Uploaded Liver Scan is Abnormal"
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)

def graph():
    height = [svm_acc, cnn_acc]
    bars = ('SVM Accuracy', 'CNN Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='Prediction of Time-to-Event Outcomes in Diagnosing Liver Cancer Based on SVM and Compare the Accuracy of Predicted Outcome with Deep CNN Algorithm')
title.config(bg='deep sky blue', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Liver Cancer Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

readButton = Button(main, text="Read & Split Dataset to Train & Test", command=splitDataset)
readButton.place(x=350, y=550)
readButton.config(font=font1)

svmButton = Button(main, text="Execute SVM Accuracy Algorithms", command=executeSVM)
svmButton.place(x=50, y=600)
svmButton.config(font=font1)

cnnButton = Button(main, text="Execute CNN Accuracy Algorithm", command=executeCNN)
cnnButton.place(x=350, y=600)
cnnButton.config(font=font1)

predictButton = Button(main, text="Predict Liver Cancer", command=predictCancer)
predictButton.place(x=50, y=650)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=350, y=650)
graphButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()