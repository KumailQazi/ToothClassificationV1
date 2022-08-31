from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
def toothClassification():
    source = os.getcwd()
    train = ImageDataGenerator(rescale=1/255)
    validation= ImageDataGenerator(rescale=1/255)

    train_dataset = train.flow_from_directory(source+"\\DentalImages\\Train",target_size =(200,200),batch_size=3,class_mode="binary")
    validaion_dataset = validation.flow_from_directory(source+"\\DentalImages\\Validate",target_size =(200,200),batch_size=2,class_mode="binary")

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(200,200,3)),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        #
                                        tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        #
                                        tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        ##
                                        tf.keras.layers.Flatten(),
                                        ##
                                        tf.keras.layers.Dense(512,activation="relu"),
                                        ##
                                        tf.keras.layers.Dense(1,activation="softmax")
                                    ])

    model.compile(loss="binary_crossentropy",
                optimizer=RMSprop(learning_rate=0.001),
                metrics=["accuracy"])

    model_fit=model.fit(train_dataset,steps_per_epoch=5,epochs=5,validation_data=validaion_dataset)

    dir_path=source+"\\DentalImages\\Test"

    for i in os.listdir(dir_path):
        img=image.load_img(dir_path+"\\"+i,target_size =(200,200))
        plt.imshow(img)
        plt.show()
        X=image.img_to_array(img)
        X=np.expand_dims(X,axis=0)
        images=np.vstack([X])
        val = model.predict(images)
        if val==0:
            print(" you are old")
        elif val==1:
            print(" you are teen")
        else:
            print(" not predicted")