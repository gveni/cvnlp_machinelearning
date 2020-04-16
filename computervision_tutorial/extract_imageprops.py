# display image properties
from PIL import Image

# load image
image_path = '/Users/gveni/Documents/data/cv_data/sydney_bridge.png' 
image = Image.open(image_path)
# extract image properties
print('Image type:', image.format)
print('Image channel-type:', image.mode)
print('Image size:', image.size)
# display image
image.show()
