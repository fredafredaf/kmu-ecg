from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def generator(train_data_dir):
  train_datagen = ImageDataGenerator(rotation_range= 4, 
                                   width_shift_range = 0.05, 
                                   shear_range=0.05,
                                   fill_mode="nearest",
                                   rescale=1./255,
                                   validation_split=0.3                 
                                   )


  train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                  follow_links= True)

  val_generator = train_datagen.flow_from_directory(directory= train_data_dir,
                                follow_links = True)
  
  return train_generator, val_generator

def make_model(output):

  base_model = InceptionV3(weights='imagenet',include_top=False)
  x = base_model.output
  x1 = GlobalAveragePooling2D()(x)
  x2 = Dropout(0.4)(x1)
  x3 = Dense(512,activation='relu')(x2)
  x4 = Dense(128,activation='relu')(x3)
  predictions = Dense(output,activation='softmax')(x4)
  model = Model(inputs=base_model.input,outputs=predictions)
  model.compile(optimizer=RMSprop(learning_rate=0.03,momentum= 0.01, epsilon=0.1, decay= 0.2),loss='categorical_crossentropy',metrics=['accuracy'])
  
  return model

