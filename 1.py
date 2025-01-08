import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn



def fill_image():
    indexes = [                          #Изображение в виде вектора/торча
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
        
    ]

    image = np.zeros(35).reshape((5,7)) #Изображение размеры подбираю ЯR

    for i in range(len(indexes)):
        image[i][indexes[i]] = 1

    return image

def Conv(im):
    core = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
        ])
    
    ans = np.zeros( (len(im) - len(core) + 1) * (len(im[0]) - len(core[0]) + 1) ).reshape( (len(im) - len(core) + 1, len(im[0]) - len(core[0]) + 1) ) #Получившийся при конвертации размер Core - Image + 1
    

    for i in range(0, len(im) - len(core) + 1, 1):
        for j in range(0, len(im[0]) - len(core[0]) + 1, 1):
            
            partedMatr = np.zeros(len(core)*len(core[0])).reshape((len(core), len(core[0]))) #Частичная матрица для совмещения с ядром. Размер как у ядра

            for row in range( len(core)):
                for col in range( len(core[0]) ):
                    partedMatr[row][col] = im[i+row][j+col]

               
            
            n = np.sum([partedMatr[x, y] * core[x, y] for x in range(len(partedMatr)) for y in range(len(partedMatr))  ])
            ans[i, j] = n
            #ans[i, j] = partedMatr.sum()
            
           
            
    
    

    return ans
    


print()
print()
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])
print(image)
print()
Converted_Image = Conv(image)
print(Converted_Image)

Count = Converted_Image.sum()
print(Count)
