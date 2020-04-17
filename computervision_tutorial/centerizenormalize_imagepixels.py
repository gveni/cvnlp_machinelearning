# standardize image pixels to make them zero-mean and unit variance (subtract from  mean and divide by standard
# deviation) and then rescale them to [0,1]
import numpy as np
from PIL import Image

image_path = '/Users/gveni/Documents/data/cv_data/sydney_bridge.png'
ip_image = Image.open(image_path)  # load input image
ip_image = np.asarray(ip_image)  # convert to numpy array
ip_image = ip_image.astype('float32')
print('Image min = %.3f, max = %.3f' %(np.min(ip_image),np.max(ip_image)))
# global standardization of image
image_mean, image_std = np.mean(ip_image), np.std(ip_image)
print('Image mean = %.3f and standard deviation = %.3f' %(image_mean, image_std))
standardized_image = (ip_image - image_mean)/image_std
image_mean, image_std = np.mean(standardized_image), np.std(standardized_image)
print('After global standardization, image mean = %.3f and standard deviation = %.3f' %(image_mean, image_std))
print('After global standardization, image min = %.3f, max = %.3f'
%(np.min(standardized_image),np.max(standardized_image)))
# clip pixels between -1 and 1
standardized_image = np.clip(standardized_image, -1.,1.)
# scale values to [0,1]
rescaled_image = (standardized_image + 1.)/2.
print('After rescaling, image mean = %.3f and standard deviation = %.3f'
%(np.mean(rescaled_image), np.std(rescaled_image)))
print('After rescaling, image min = %.3f, max = %.3f'%(np.min(rescaled_image),
np.max(rescaled_image)))
