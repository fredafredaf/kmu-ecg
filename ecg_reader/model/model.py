
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Flatten, TimeDistributed,GlobalAveragePooling2D, LSTM, MaxPool2D, Dense, Dropout, Input, Conv2D, BatchNormalization
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.activations import relu, softmax


def pretrained_model(output):

  base_model = InceptionV3(weights='imagenet',include_top=False)
  x = base_model.output
  x1 = GlobalAveragePooling2D()(x)
  x2 = Dropout(0.4)(x1)
  x3 = Dense(512,activation='relu')(x2)
  x4 = Dense(128,activation='relu')(x3)
  predictions = Dense(output,activation='softmax')(x4)

  for layer in base_model.layers:
    #the default is true. we want to first focus on the head
	  layer.trainable = False

  model = Model(inputs=base_model.input, outputs=predictions)
  model.compile(optimizer=RMSprop(learning_rate=0.03,momentum= 0.01, epsilon=0.1, decay= 0.2),loss='categorical_crossentropy',metrics=['accuracy'])
  
  return model

def CNN_LSTM(output):

  cnn = Sequential([
      Conv2D(filters=128, kernel_size=20, strides=3, padding='same',activation=relu),
      BatchNormalization(),
      MaxPool2D(pool_size=2, strides=3),
      Conv2D(filters=32, kernel_size=7, strides=1, padding='same', activation=relu),
      BatchNormalization(),
      MaxPool2D(pool_size=2, strides=2),
      Conv2D(filters=32, kernel_size=10, strides=1, padding='same', activation=relu),
        # tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
      MaxPool2D(pool_size=2, strides=2),
      Flatten(),
        # tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
        # tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
    ])

  model= Sequential([
      TimeDistributed(cnn)
      LSTM(10, input_shape=(360,1)),
      Flatten(),
        # tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
      Dropout(rate=0.1),
      Dense(units=20, activation= relu),
      Dense(units=10, activation= relu),
      Dense(units=output, activation= softmax)
  ])
  
  model.compile(optimizer=RMSprop(learning_rate=0.03,momentum= 0.01, epsilon=0.1, decay= 0.2),loss='categorical_crossentropy',metrics=['accuracy'])
  
  return model

