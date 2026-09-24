from PIL import Image
import numpy as np

img = np.zeros((224, 224, 3), dtype=np.uint8)
img[100:150, 100:150, 0] = 255
Image.fromarray(img).save('test.png')
