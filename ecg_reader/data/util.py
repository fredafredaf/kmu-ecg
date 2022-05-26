from PIL import Image
from matplotlib import pyplot as plt
import random
import os


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