import gdown
import zipfile
import os
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import shutil
from pathlib import Path
import random
import numpy as np

DIRNAME = Path('./data')

def download_unzip_data():
  url_1 = "https://drive.google.com/uc?id=1odCzBSQI4I--kLz2gVrRVumNIZyMnRwn"
  url_2 = "https://drive.google.com/uc?id=1vHVA1clstiElzIEHZx2oYdAoVwvNC1b8"

  DIRNAME.mkdir(parents=True, exist_ok=True)

  file_dir_acs = DIRNAME / "ACS.zip"
  file_dir_norm = DIRNAME / "norm.zip"

  data_dir = DIRNAME/ "acs"

  #if the file is already downloaded then skip the following
  if file_dir_acs.exists() or data_dir.exists():
      print(f"Data Already here!")
      return 
      
  print(f"Downloading raw dataset from {url_1} to {file_dir_acs}...")
  gdown.download(url_1, str(file_dir_acs), quiet=True)
  print(f"Downloading raw dataset from {url_2} to {file_dir_norm}...")
  gdown.download(url_2, str(file_dir_norm), quiet=True)

  for n, f in zip(["acs", "norm"], [file_dir_acs, file_dir_norm]):
    unzipped_files = DIRNAME/ n
    unzipped_files.mkdir(parents=True, exist_ok=True)

    #unzipped all files into data/acs/files and data/norm/files
    with zipfile.ZipFile(str(f), 'r') as zip_ref:
      zip_ref.extractall(str(unzipped_files)+ '/files')


def sort():
  for n in ["acs", "norm"]:
    unzipped_files = DIRNAME/ n

    for t in ["ecg", "sound", "sound_img"]:
      im_types = unzipped_files / t
      im_types.mkdir(parents=True, exist_ok=True)

    path = DIRNAME/ n / 'files'

    #use the first image to get shape to segment the image
    img = plt.imread(path/ os.listdir(str(path))[0])
    h, w, c = img.shape
    # calculate a block (grid)'s length
    block = round(w/50) 
  
    for f in os.listdir(str(path)):
      if f[-5] == "L":
        img = plt.imread(path / f)
        # remove_text 
        img = np.array(Image.fromarray(img[ :, block*2: , :]))
      
        ecg = Image.fromarray(img[ 0:block*7, : , :])
        sound = Image.fromarray(img[ block*7:block*12, : , :])
        sound_img = Image.fromarray(img[ block*12:, : , :])

        ecg.save('./data/' + n + '/ecg/' + f[:-5]+ ".jpg")
        sound.save('./data/' + n + '/sound/' + f[:-5] + ".jpg")
        sound_img.save('./data/' + n + '/sound_img/' + f[:-5] + ".jpg")

    data_num(n)

def create_dir():

  dirname = Path('./data')

  split = dirname/ 'split_data'
  split.mkdir(parents=True, exist_ok=True)

  for i in ['train', 'val', 'test']:
    split_data = split/ i
    split_data.mkdir(parents=True, exist_ok=True)

    for j in ['ecg', 'sound', 'sound_img']:
      im_type = split_data/ j
      im_type.mkdir(parents=True, exist_ok=True)

      for k in ['acs', 'norm']:
        im_class = im_type/ k
        im_class.mkdir(parents=True, exist_ok=True)

def data_num(file):
  li = os.listdir('./data/'+ file + '/ecg/') # dir is your directory path
  num_files = len(li)
  print(f"The number of patients with {file} is {num_files}")


def remove_dir():
  for i in ['train', 'val', 'test']:
    shutil.rmtree('./data/split_data'+ i)

def prepare_data():
  download_unzip_data()
  sort()
  create_dir()

  #return './data/split_data/'
  return 'ecg_reader/data'

if __name__ == "__main__":
  prepare_data()

#--------------------------------------------------------------------------#

def visualize(im_class, im_type, num =10):
  print(f'Display Random {im_type} images of {im_class}')
  # Adjust the size of your images
  plt.figure(figsize=(40, 20))
  img_dir = './data/' + im_class + '/'+ im_type + '/'
  random_numbers = random.sample(range(len(os.listdir(img_dir))), num)

  # Iterate and plot random images
  for i,j in zip(range(num), random_numbers):
    plt.subplot(num/2, 2, i + 1)
    img = plt.imread(img_dir + os.listdir(img_dir)[j])
    plt.imshow(img, cmap='gray')
    plt.axis('off')   
  # Adjust subplot parameters to give specified padding
  plt.tight_layout() 

def move_data(dist_acs, dist_norm):

  for i in ['acs', 'norm']:

    for j in ['ecg', 'sound', 'sound_img']:
      path = './data/' + i + '/'+ j + '/'
      file_name = os.listdir(path)

      if i == 'acs':
        for k in dist_acs[0]:
          shutil.copy(path + file_name[k], './data/split_data/train/'+ j +'/acs/' + file_name[k])

        for k in dist_acs[1]:
          shutil.copy(path + file_name[k], './data/split_data/val/'+ j +'/acs/' + file_name[k])

        for k in dist_acs[2]:
          shutil.copy(path + file_name[k], './data/split_data/test/'+ j +'/acs/' + file_name[k])

      elif i == 'norm':
        for k in dist_norm[0]:
          shutil.copy(path + file_name[k], './data/split_data/train/'+ j +'/norm/' + file_name[k])

        for k in dist_norm[1]:
          shutil.copy(path + file_name[k], './data/split_data/val/'+ j +'/norm/' + file_name[k])

        for k in dist_norm[2]:
          shutil.copy(path + file_name[k], './data/split_data/test/'+ j +'/norm/' + file_name[k])

def split_num(seed, train_pr = 0.7, if_print = True):

  acs_num = len(os.listdir('./data/acs/ecg'))
  train_num_acs = int(acs_num* train_pr)
  test_num_acs = int((acs_num- train_num_acs)/2)
  val_num_acs = acs_num - test_num_acs - train_num_acs

  norm_num = len(os.listdir('./data/norm/ecg'))
  train_num_norm = train_num_acs * 2
  test_num_norm = test_num_acs
  val_num_norm = val_num_acs

  if if_print:

    print()
    print(f"Num. of train in norm is {train_num_norm}")
    print(f"Num. of train in acs is {train_num_acs}")
    print(f"Total train: {train_num_norm+ train_num_acs}")

    print()
    print(f"Num. of val in norm is {val_num_norm}")
    print(f"Num. of val in acs is {val_num_acs}")
    print(f"Total val: {val_num_norm + val_num_acs}")

    print()
    print(f"Num. of ctr in test is {test_num_norm}")
    print(f"Num. of exp in test is {test_num_acs}")
    print(f"Total test: {test_num_norm + test_num_acs}")

    np.random.seed(seed)

    dist_acs = np.arange(0, acs_num, 1)
    chunk_size = [train_num_acs, val_num_acs, test_num_acs]
    dist_acs = [np.random.choice(dist_acs,_, replace=False) for _ in chunk_size]

    dist_norm = np.arange(0, norm_num, 1)
    chunk_size = [train_num_norm, val_num_norm, test_num_norm]
    dist_norm = [np.random.choice(dist_norm,_, replace=False) for _ in chunk_size]

  return dist_acs, dist_norm

def dir_for_generator():

  for i in ['ecg', 'sound', 'sound_img']:
    DIR = DIRNAME / i
    DIR.mkdir(parents=True, exist_ok=True)
    
    for j in ['acs', 'norm']:
      DIR_2 = DIR / j
      DIR_2.mkdir(parents=True, exist_ok=True)

      source_folder = './data/'+ j + '/' + i + '/'
      destination_folder = './data/'+ i + '/' + j + '/'
      for file_name in os.listdir(source_folder):

        source = source_folder + file_name
        destination = destination_folder + file_name
    # move only files
        if os.path.isfile(source):
          shutil.copy(source, destination)
        



 