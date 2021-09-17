import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def hough(l):
    for item in l:
        img=cv2.imread (item)
        gray=cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny (gray, 75, 150)
        lines=cv2.HoughLinesP (edges, 1, np.pi/180, 30, maxLineGap=250)
        for line in lines:
            x1, y1, x2, y2=line [0]
            cv2.line (img, (x1, y1), (x2, y2), (0, 0, 128), 1)
        print(len(lines))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow (edges,cmap="gray"), plt.show ()
        plt.imshow(img),plt.show()

if __name__ == '__main__':

    imgs=Path(r'C:\Users\tomsk\Desktop\alghoritms\images')
    l=[]
    for item in list (imgs.iterdir()):
        item=item.as_posix()
        l.append(item)

    while True:
        res=input('Type number:\n'
                    '1 - exit-0 - exit-32 - exit-64\n'
                    '2 - exit-96 - exit-128 - exit-160\n'
                    '3 - exit-192 - exit-224 - exit-255\n'
                    'Exit: (Y/y)\n')

        data = {'1':[l[0:3]],
                '2':[l[3:6]],
                '3':[l[6:9]]}


        for k,v in data.items():
            if k == res:
                hough(*v)

        if res == 'Y' or res == 'y':
            exit()
        else:
            print('Allowed numbers: 1,2,3')