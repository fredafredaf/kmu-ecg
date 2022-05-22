
import gdown
import zipfile
import os
import shutil
from PIL import Image
#from matplotlib import pyplot as plt
import shutil
from pathlib import Path



dirname = Path('./data')

def download_data(dirname: Path) -> Path:
  url_1 = "https://drive.google.com/uc?id=1odCzBSQI4I--kLz2gVrRVumNIZyMnRwn"
  url_2 = "https://drive.google.com/uc?id=1vHVA1clstiElzIEHZx2oYdAoVwvNC1b8"

  dirname.mkdir(parents=True, exist_ok=True)

  filename_1 = dirname / "ACS.zip"
  filename_2 = dirname / "norm.zip"

  if filename_1.exists() and filename_2.exists():
      return [filename_1, filename_2]

  print(f"Downloading raw dataset from {url_1} to {filename_1}...")
  gdown.download(url_1, str(filename_1), quiet=True)
  print(f"Downloading raw dataset from {url_2} to {filename_2}...")
  gdown.download(url_2, str(filename_2), quiet=True)

  return [filename_1, filename_2]

def unzip(dirname: Path, filenames: Path):  #filename is either acs or norm

  for n, f in zip(["acs", "norm"], filenames):
  
    dir_1 = dirname/ n
    dir_1.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(f), 'r') as zip_ref:
      zip_ref.extractall(str(dir_1)+ '/files')

    for t in ["ecg", "sound", "sound_img"]:

      dir_2 = dir_1 / t
      dir_2.mkdir(parents=True, exist_ok=True)

def sort():

  for n in ["acs", "norm"]:
    path = './data/'+ n + '/files/'

    #use the first image to get shape to trim the image
    img = plt.imread(path + os.listdir(path)[0])
    h, w, c = img.shape
    block = round(w/50) #a block (grid)'s length
  

    for f in os.listdir(path):
      if f[-5] == "L":
        img = plt.imread(path + f)

        # remove_text = Image.fromarray(img[ :, block*2: , :])
        

        ecg = Image.fromarray(img[ 0:block*7, : , :])
        sound = Image.fromarray(img[ block*7:block*12, : , :])
        sound_img = Image.fromarray(img[ block*12:, : , :])

        ecg.save('./data/'+ n + '/ecg/' + f[:-5] + ".jpg")
        sound.save('./data/'+ n + '/sound/' + f[:-5] + ".jpg")
        sound_img.save('./data/'+ n + '/sound_img/' + f[:-5] + ".jpg")

def data_num(file):

  li = os.listdir('./data/'+ file + '/ecg/') # dir is your directory path
  num_files = len(li)
  print(f"The number of patients with {file} is {num_files}")

'''
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
'''

def create_dir():

  dirname = Path('./data')

  direct = dirname/ 'train'
  direct.mkdir(parents=True, exist_ok=True)

  for j in ['ecg', 'sound', 'sound_img']:
    director = direct/ j
    director.mkdir(parents=True, exist_ok=True)

    for k in ['acs', 'norm']:
      directory = director/ k
      directory.mkdir(parents=True, exist_ok=True)

def move_data():

  for i in ['acs', 'norm']:

    for j in ['ecg', 'sound', 'sound_img']:
      path = './data/' + i + '/'+ j + '/'
      file_name = os.listdir(path)

      target = './data/train/' + j + '/'+ i + '/'

      for f in file_name:
        shutil.copy(os.path.join(path, f), target)

def prepare_data():

  file = download_data(dirname)
  unzip(dirname, file)
  sort()
  create_dir()
  move_data()

  return './data/train/'

 