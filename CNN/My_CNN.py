import torch

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x**2 - torch.sin(x) + 5.


# здесь объявляйте необходимые функции
def model(x, w):
    return x @ w

def loss(x, w, y):
    return (model(x,w) - y)

def Q(x, w, y):
    return torch.mean(loss(x, w, y)**2)

def dQ(x, w, y):
    return 2 * torch.mean( loss(x,w,y) @ x.T ) 

coord_x = torch.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

X = torch.stack([torch.ones_like(coord_x), coord_x, coord_x**2, coord_x**3], dim=1)

sz = coord_x.size(0) # количество значений функций (точек)
eta = torch.tensor([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = torch.zeros(4, dtype=torch.float32) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма
sz = len(X)
# здесь продолжайте программу


for i in range(N):
    
    a = model(X, w)  
    err = a - coord_y
    grad = 2 / sz * (err @ X)
    w -= eta * grad

Qe = Q(X, w, coord_y)

print(w)
print(Qe)