import numpty as np

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

print()
print()
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

core = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
        ])

print(image)
print()
Converted_Image = Conv(image, core)
print(Converted_Image)

Count = Converted_Image.sum()
print(Count)