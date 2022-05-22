import shutil
import os
from pathlib import Path
from training import model


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

'''
def create_dir():

  dirname = Path('./data')

  for i in ['train', 'val', 'test']:
    direct = dirname/ i 
    direct.mkdir(parents=True, exist_ok=True)

    for j in ['ecg', 'sound', 'sound_img']:
      director = direct/ j
      director.mkdir(parents=True, exist_ok=True)

      for k in ['acs', 'norm']:
        directory = director/ k
        directory.mkdir(parents=True, exist_ok=True)
'''

def remove_dir():

  for i in ['train', 'val', 'test']:
    shutil.rmtree('./data/'+ i)


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
    print(f"Total test: {val_num_norm + val_num_acs}")

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

def move_data(dist_acs, dist_norm):

  for i in ['acs', 'norm']:

    for j in ['ecg', 'sound', 'sound_img']:
      path = './data/' + i + '/'+ j + '/'
      file_name = os.listdir(path)

      if i == 'acs':
        for k in dist_acs[0]:
          shutil.copy(path + file_name[k], './data/train/'+ j +'/acs/' + file_name[k])

        for k in dist_acs[1]:
          shutil.copy(path + file_name[k], './data/val/'+ j +'/acs/' + file_name[k])

        for k in dist_acs[2]:
          shutil.copy(path + file_name[k], './data/test/'+ j +'/acs/' + file_name[k])

      elif i == 'norm':
        for k in dist_norm[0]:
          shutil.copy(path + file_name[k], './data/train/'+ j +'/norm/' + file_name[k])

        for k in dist_norm[1]:
          shutil.copy(path + file_name[k], './data/val/'+ j +'/norm/' + file_name[k])

        for k in dist_norm[2]:
          shutil.copy(path + file_name[k], './data/test/'+ j +'/norm/' + file_name[k])
