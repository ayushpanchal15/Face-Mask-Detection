import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths


Initial_Learning_Rate = 1e-5
PASS = 15
Batch_Size = 49

DIRECTORY = r"D:\custom_data_numberplate\custom_data_numberplate"
CATEGORIES = ["custom_data_numberplate"]


Data_Sets = [] # Data_Set
labels = []

for cat in CATEGORIES:
    path = os.path.join(DIRECTORY, cat)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	Data_Sets.append(image)
    	labels.append(cat)

Label_Binarizer = LabelBinarizer() #Label_Binarizer
labels = Label_Binarizer.fit_transform(labels)
labels = to_categorical(labels)

Data_Sets = np.array(Data_Sets, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(Data_Sets, labels,	test_size=0.15, stratify=labels, random_state=30)


aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.20,
	width_shift_range=0.3,
	height_shift_range=0.3,
	shear_range=0.17,
	horizontal_flip=True,
	fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))



headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(138, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
	layer.trainable = False


# optimization
optimize_model = Adam(lr=Initial_Learning_Rate, decay=Initial_Learning_Rate / PASS)
model.compile(loss="binary_crossentropy", optimizer=optimize_model,
	metrics=["accuracy"])
# training_head

training_head = model.fit(
	aug.flow(trainX, trainY, batch_size=Batch_Size),
	steps_per_epoch=len(trainX) // Batch_Size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // Batch_Size,
	epochs=PASS)


predIdxs = model.predict(testX, batch_size=Batch_Size)

predIdxs = np.argmax(predIdxs, axis=1)


print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=Label_Binarizer.classes_))



model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
Number = PASS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, Number), training_head.history["loss"], label="train_loss")
plt.plot(np.arange(0, Number), training_head.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, Number), training_head.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, Number), training_head.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Pass #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
