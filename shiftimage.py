import cv2
import numpy as np

letter = 'B'

img_path = 'images/' + letter + '/' + letter + '1.png'
img = cv2.imread(img_path, 0) # read image as grayscale. Set second parameter to 1 if rgb is required

img2 = np.copy(img)

#a: shift right
#b: shift down
for a in range(0,9):
    for b in range(0,8):
        #whiten everything left and above the letter
        for i in range(0,20):
            for j in range(0,a):
                img2[i][j] = 255
            for k in range(0,b):
                img2[k][i] = 255
        #shift
        for x in range(0,20):
            for y in range(a,20):
                img2[x][y] = img[x-b][y-a]
        cv2.imwrite('images/' + letter + '/' + letter + str(a+1+9*b) + '.png', img2)
