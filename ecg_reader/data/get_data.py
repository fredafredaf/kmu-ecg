import gdown
import os
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import zipfile
import random

def download_data():
  url_1 = "https://drive.google.com/uc?id=1odCzBSQI4I--kLz2gVrRVumNIZyMnRwn"

  if not os.path.exists('./data/'):
    os.makedirs('./data/')

  output = './data/acs.zip'
  gdown.download(url_1, output, quiet=False)

  url_2 = "https://drive.google.com/uc?id=1vHVA1clstiElzIEHZx2oYdAoVwvNC1b8"
  output = './data/norm.zip'
  gdown.download(url_2, output, quiet=False)

def unzip(file):  #file is either acs or norm

  if not os.path.exists('./data/' + file):
    os.makedirs('./data/' + file)

  if not os.path.exists('./data/'+ file + '/ecg'):
    os.makedirs('./data/'+ file + '/ecg')

  if not os.path.exists('./data/'+ file + '/sound'):
    os.makedirs('./data/'+ file + '/sound')

  if not os.path.exists('./data/'+ file + '/sound_img'):
    os.makedirs('./data/'+ file + '/sound_img')

  if not os.path.exists('./data/'+ file + '/files'):
    os.makedirs('./data/'+ file + '/files')   

  file_name = './data/' + file + '.zip'

  with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall('./data/'+ file + '/files')

def sort(file):

  path = './data/'+ file + '/files/'

  #use the first image to get shape to trim the image
  img = plt.imread(path + os.listdir(path)[0])
  h, w, c = img.shape
  block = round(w/50) #a block (grid)'s length
  

  for f in os.listdir(path):
    if f[-5] == "L":
      img = plt.imread(path + f)

      #remove_text = Image.fromarray(img[ :, block*2: , :])

      ecg = Image.fromarray(img[ 0:block*7, : , :])
      sound = Image.fromarray(img[ block*7:block*12, : , :])
      sound_img = Image.fromarray(img[ block*12:, : , :])

      ecg.save('./data/'+ file + '/ecg/' + f[:-5] + ".jpg")
      sound.save('./data/'+ file + '/sound/' + f[:-5] + ".jpg")
      sound_img.save('./data/'+ file + '/sound_img/' + f[:-5] + ".jpg")

def data_num(file):

  list = os.listdir('./data/'+ file + '/ecg/') # dir is your directory path
  num_files = len(list)
  print(f"The number of patients with {file} is {num_files}")

def visualize(file, im_type):
  print(f'Display Random {im_type} images of {file}')

  # Adjust the size of your images
  plt.figure(figsize=(40, 20))
  num = 10
  img_dir = './data/' + file + '/'+ im_type + '/'
  random_numbers = random.sample(range(len(os.listdir(img_dir))), num)


  # Iterate and plot random images
  for i,j in zip(range(num), random_numbers):
    plt.subplot(num/2, 2, i + 1)
    img = plt.imread(img_dir + os.listdir(img_dir)[j])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
  # Adjust subplot parameters to give specified padding
  plt.tight_layout() 

if __name__ == "__main__":
  
  download_data()
  unzip('acs')
  unzip('norm')
  sort('acs')
  sort('norm')

  print()
  data_num("acs")
  data_num("norm")

 