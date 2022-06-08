#ttimport wandb
import argparse
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
import numpy as np
#from wandb.keras import WandbCallback

from pathlib import Path
import sys
print(sys.path)

from importlib.util import find_spec
if find_spec("ecg_reader") is None:
    import sys
    sys.path.append('..')

from ecg_reader.data.get_data import prepare_data
from ecg_reader.model.model import CNN_LSTM, pretrained_model

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

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--im_type', type=str, default = 'ecg')
  parser.add_argument('--epochs', type=int, default = 10)
  parser.add_argument('--model', type=str, default= CNN_LSTM )
  return parser


def main():
  parser = get_parser()
  args = parser.parse_args()

  dirname = prepare_data()
  train_dir = dirname + args.im_type 
  train_gen, val_gen = generator(train_dir)

  class_weights = dict(zip(np.unique(train_gen.classes), class_weight.compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(train_gen.classes),
                                            y= train_gen.classes)))
  
  model = CNN_LSTM(2)
  history = model.fit(train_gen,
            batch_size = 16,
            epochs= args.epochs, 
            validation_data= val_gen,
            class_weight=class_weights
          )

  type_and_epoch = args.im_type + "_" + str(args.epochs)
  dirname = Path('./saved_model')/ type_and_epoch
  dirname.mkdir(parents=True, exist_ok=True)

  model.save(str(dirname))

if __name__ == "__main__":
  main()


'''
def main():
  parser = get_parser()
  args = parser.parse_args()

  for run in range(5):
    # Start a run, tracking hyperparameters
    wandb.init(
        project="5 data comb",
        # Set entity to specify your username or team name
        # ex: entity="wandb",
        config={
        })
    config = wandb.config

    # Get the data
    create_dir()
    dist_acs, dist_norm = split_num(run)
    move_data(dist_acs, dist_norm)

    model = make_model(2)
    train_dir = './data/train/' +  args.im_type 
    val_dir = './data/val/' +  args.im_type 
    test_dir = './data/test/' +  args.im_type 
    train_generator = train_gen(train_dir)
    val_generator = val_gen(val_dir)

    class_weights = class_weight.compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(train_generator.classes),
                                            y= train_generator.classes)
    class_weights = dict(zip(np.unique(train_generator.classes), class_weights))

    # WandbCallback auto-saves all metrics from model.fit(), plus predictions on validation_data
    logging_callback = WandbCallback(log_evaluation=True)

    history = model.fit(train_generator,
                        batch_size = 16,
                        epochs= arg.epochs, 
                        validation_data= val_generator,
                        class_weight=class_weights,
                        callbacks=[logging_callback]
                        )
    
    test_results.append(test_result(test_dir, model))

    remove_dir()
    # Mark the run as finished
    wandb.finish()

  '''

