import numpy as np
from PIL import Image
import glob


img = Image.open('/home/charu/mnist/test/0/00003.png').convert('RGBA')

arr = np.array(img)

shape = arr.shape

flat_arr = arr.ravel()



print len(flat_arr)/3
#vector = np.matrix(flat_arr)


for infile in glob.glob("/home/charu/mnist/test/0/*.png"):
    print infile