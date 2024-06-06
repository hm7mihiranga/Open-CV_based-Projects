```diff
I used google colab for the development, so the Kaggle dataset getting through the google collaborator. 
-!pip install Kaggle

This is used to getting access for the google drive
-from google.colab import drive
-drive.mount('/content/drive')

Make the directory on the colab notebook
-! mkdir ~/.kaggle

This copies the Keggle.Json file from the specified location
-!cp /content/drive/MyDrive/keggal_Api_json/kaggle.json ~/.kaggle/kaggle.json

This command sets the permissions of the kaggle.json file to read and write only for the owner (you). The 600 mode ensures that only you can access the file.
-! chmod 600 ~/.kaggle/kaggle.json

This downloads the Kaggle dataset with the ID paultimothymooney/chest-xray-pneumonia
-!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

This used to extract the zip file
-! unzip chest-xray-pneumonia.zip

This is used to save the Tensorflow
-!pip install tensorflow

SciPy provides a wide range of functions for optimization, linear algebra, signal processing, and more. 
-!pip install scipy

In here I used VGG16 preprocessing model to classified and recognize the images.
-from tensorflow import keras
-from keras.layers import Input, Lambda, Dense, Flatten
-from keras.models import Model
-from keras.applications.vgg16 import VGG16
-from keras.applications.vgg16 import preprocess_input
-from keras.preprocessing import image

imports the ImageDataGenerator class from the Keras library. This class is commonly used for data augmentation and preprocessing when working with image data in deep learning models.
-from keras.preprocessing.image import ImageDataGenerator

imports the Sequential class from the Keras library. The Sequential model is a linear stack of layers, where you can add one layer at a time.
-from keras.models import Sequential

-import numpy as np
-from glob import glob
-import matplotlib.pyplot as plt


-vgg = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
 disregard the final fully connected layer, which does the classification. We do this, in situations, if we need to customize
 the final outputs. Usually, VGG 16 has close to 1000 output classes. But here for pneumonia prediction we need only two classes
 as positive and negative.

 we use this strategy for situations, where we need the embeddings generated from the model to be plugged with another
 customized requirements.


 prevent weights adjustments of the VGG modle during the training procedure. So it will have the pretrained imagenet weights.
-for layer in vgg.layers:
-    layer.trainable = False


-folders = glob('/content/chest_xray/chest_xray/train/*')
-x = Flatten()(vgg.output) # flatten the embeddings generated to one single tensor, overl the 1000s of tensors output


 use the derived flattened tensor and re-use it to create a custom and trainable dense layer for us to get the binary class
 output we need as penumonia positive / negative. Hence it`s binary classed, we can use softmax either.
-prediction = Dense(len(folders), activation='softmax')(x)
 create a model object
-model = Model(inputs=vgg.input, outputs=prediction)
 view the structure of the model
-model.summary()

 configure the back prop pipiline
-model.compile(
-  loss='categorical_crossentropy',
-  optimizer='adam',
-  metrics=['accuracy']
-)


 used for the image augmentation process.
-from keras.preprocessing.image import ImageDataGenerator


 augmentation pipeline
-train_datagen = ImageDataGenerator(rescale = 1./255,
-                                   shear_range = 0.2,
-                                   zoom_range = 0.2,
-                                   horizontal_flip = True)

-test_datagen = ImageDataGenerator(rescale = 1./255)



 Make sure you provide the same target size as initialied for the image size
 batch size means, size of images you take from the dataset for one backprop training.
 one epoc has multiple batches. Untill the dataset size is reached , batches are taken inside the given epoc.
 once the entire datset is taken inside, first epoc will be completed. Then repeat the same process for the other epocs as well
-training_set = train_datagen.flow_from_directory('/content/chest_xray/train',
-                                                 target_size = (224, 224),
-                                                 batch_size = 10,
-                                                 class_mode = 'categorical')



-test_set = test_datagen.flow_from_directory('/content/chest_xray/test',
-                                            target_size = (224, 224),
-                                            batch_size = 10,
-                                            class_mode = 'categorical')

 training procedure
-r = model.fit_generator(
-  training_set,
-  validation_data=test_set,
-  epochs=30,
-  steps_per_epoch=len(training_set),
-  validation_steps=len(test_set)
-)



-import tensorflow as tf
-from keras.models import load_model

-model.save('/content/drive/MyDrive/keggal_Api_json/chest_xray.h5') # save the trained modle


-import tensorflow as tf
-from keras.models import load_model
-from keras.preprocessing import image
-from keras.applications.vgg16 import preprocess_input # import the model`s skeleton
-import numpy as np
-model=load_model('/content/drive/MyDrive/keggal_Api_json/chest_xray.h5')


load the test image
tf.keras.utils.load_img
-img=tf.keras.utils.load_img('/content/drive/MyDrive/keggal_Api_json/download.jpg',target_size=(224,224))

-x=tf.keras.preprocessing.image.img_to_array(img) # image as a numpy array

-x=np.expand_dims(x, axis=0)

-img_data=preprocess_input(x) # organize for the prediction


-classes=model.predict(img_data)

-result=int(classes[0][0])


-if result== 0:
-    print("Person is Affected By PNEUMONIA")
-else:
-    print("Result is Normal")
