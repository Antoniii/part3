# coding: utf8

'''
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy


def makeData ():
    x = numpy.arange (-10, 10, 0.1)
    y = numpy.arange (-10, 10, 0.1)
    xgrid, ygrid = numpy.meshgrid(x, y)

    zgrid  = (xgrid)**2 + (ygrid)**2
    return xgrid, ygrid, zgrid

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)

axes.plot_surface(x, y, z)

pylab.show()



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5.2, 5.2, 0.25)
Y = np.arange(-5.2, 5.2, 0.25)
X, Y = np.meshgrid(X, Y)
R = 0.95*np.sqrt(X**2 + Y**2)
Z = np.cos(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

'''

"""
from pylab import *
'''
plot(range(1, 20),
     [i**3 for i in range(1, 20)], 'ro')
#savefig('example.png')
show()
'''

plot([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8], 
     [0.5,1,1,1,1.5,1.2,1.1,1,-0.5,-1,-1,-1,-1.5,-1.2,-1.1,-1],
     'b|',     markersize = 15.25) 
axis([0, 80, -2, 2])

plt.xlabel('z, cm')    # обозначение оси абсцисс
plt.ylabel('r, mm')    # обозначение оси ординат

plt.title('Without space charge: P_input = 10 mWt, P_output = 300 Wt')

#plt.grid(True)
#savefig('oldgrupelectron.png')
show()
"""

'''
# Специальные функции

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import *
from pylab import *

z = 0.0
z1 = 1.0
x = linspace(0.01, 4)
y = iv(z,x)
y_1 = iv(z1,x) 
# !!! Включить оформление в стиле xkcd.com
#plt.xkcd()

plt.plot (x, y, label = 'y = I_0(x)')
plt.plot(x, y_1, label = 'y = I_1(x)')
#csfont = {'fontname':'Comic Sans MS'}
#hfont = {'fontname':'Helvetica'}

plt.title('Modified Bessel function of the first kind of real order')
#plt.xlabel('x label', **csfont)
#plt.xlabel('x')    # обозначение оси абсцисс
#plt.ylabel('y = x - eln(x)')    # обозначение оси ординат
plt.grid(True)
plt.legend(             # вставка легенды (текста в label)
    loc='upper left')    # положение легенды       
plt.show()


z0 = 0.0
z1 = 1.0
z2 = 2.0
x = linspace(0.001, 20)
y0 = jv(z0,x)
y1 = jv(z1,x)
y2 = jv(z2,x)

# !!! Включить оформление в стиле xkcd.com
#plt.xkcd()

plt.plot(x, y0, 'g^',    # маркеры из зеленых треугольников
         x, y1, 'b--',   # синяя штриховая
         x, y2, 'ro-')   # красные круглые маркеры, 
                         # соединенные сплошной линией

plt.title('Bessel function of the first kind of real order')
plt.grid(True)

plt.legend(['y = J_0(x)',
            'y = J_1(x)',
            'y = J_2(x)'])    # список легенды
          
plt.show()


z0 = 0.0
z1 = 1.0
z2 = 2.0
x = linspace(0.1, 20, 100)
y0 = yv(z0,x)
y1 = yv(z1,x)
y2 = yv(z2,x)

# !!! Включить оформление в стиле xkcd.com
#plt.xkcd()

plt.plot(x, y0, 'g^-',    # маркеры из зеленых треугольников
                         # соединённых сплошной
         x, y1, 'b--',   # синяя штриховая
         x, y2, 'ro-')   # красные круглые маркеры, 
                         # соединенные сплошной линией

axis([0, 20, -2, 1]) # размер по осям

plt.title('Bessel function of the second kind of real order')
plt.grid(True)

plt.legend(['y = Y_0(x)',
            'y = Y_1(x)',
            'y = Y_2(x)'])    # список легенды
          
plt.show()
'''




# построение графиков в 2-х разных окнах

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import *
from pylab import *

z0 = 0.0
z1 = 1.0
z2 = 2.0

x = linspace(0.001, 20)

y0 = jv(z0,x)
y1 = jv(z1,x)
y2 = jv(z2,x)

y_0 = yv(z0,x)
y_1 = yv(z1,x)
y_2 = yv(z2,x)


subplot (2, 1, 1)

plt.grid(True)

plt.plot(x, y0, 'b',    
         x, y1, 'c',   
         x, y2, 'g')    

plt.xlabel('x')    # обозначение оси абсцисс
plt.ylabel('y')    # обозначение оси ординат

#plt.title('Bessel function')

text(7, 0.85,'Bessel function', fontsize = 16)

subplot (2, 1, 2)

plt.xkcd()

axis([0, 20, -2, 1]) # размер по осям

plt.plot(x, y_0, 'k',    
         x, y_1, 'm',   
         x, y_2, 'r')

plt.xlabel('x')    # обозначение оси абсцисс
plt.ylabel('y')    # обозначение оси ординат

#plt.title('Bessel function of the second kind')

plt.legend(['y = Y_0(x)',
            r'$y = Y_1(x)$', # LaTeX формулы в легенде
            'y = Y_2(x)'],
           loc='down right')    # список легенды

text(5, 0.7,'Bessel function of the second kind', fontsize = 16)

text(4,-1.42,r"$E = mc^2 = \sqrt{{m_0}^2c^4 + p^2c^2}$") # добавление формул в стиле LaTeX

subplots_adjust(hspace=.5) # изменяет расстояние между окнами графиков

plt.show()

"""

# Список случайных округлённых величин, отсортированных по возрастанию
"""
'''
round(number[, ndigits]) - округляет число number до ndigits знаков после
запятой
(по умолчанию, до нуля знаков, то есть, до ближайшего целого)
'''
import random

high = 55.0 # - верхняя граница
low = 0.1 # - нижняя граница

def randomList(n):
    s = [0]*n
    for i in range(n):
        s[i] = low + (high-low)*random.random()
        s[i] = round(s[i],2)
    return s

a = randomList(5)
print a # до сортировки

a.sort() # Сортировка элементов массива методом .sort() производится
        # по умолчанию лексикографически — проще говоря,
        # в алфавитном порядке, а также от меньшего значения к большему.

print a # после сортировки

#print randomList(7)


def listconst(m):
    s1 = [0]*m
    for i in range(m):
        s1[i] = 1
    return s1

print listconst(5)
"""





# Элементы функционального программирования
"""
from operator import add, mul 
print add(2, mul(3, 4))

# Функция одного аргумента:
def swapcase(s): 
    return s.swapcase() 
 
print swapcase("ABC") 

# Функция двух аргументов, один из которых необязателен и имеет
# значение по умолчанию: 
def inc(n, delta=1): 
    return n+delta 
 
print inc(12) 
print inc(12, 2)


# Функция произвольного числа аргументов: 
def max_min(*args): 
  # args - список аргументов в порядке их указания при вызове 
  return max(args), min(args) 
 
print max_min(1, 2, -1, 5, 3)

'''
Пример, в котором в качестве значения по умолчанию аргумента функции 
используется изменчивый объект (список). Этот объект - один и тот же для всех 
вызовов функций, что может привести к казусам:
'''
def mylist(val, lst=[]): 
  lst.append(val) 
  return lst 
 
print mylist(1)
print mylist(2)
'''
Вместо ожидаемого [1] [2] получается [1] [1, 2], так как добавляются 
элементы к "значению по умолчанию". 
Правильный вариант решения будет, например, таким: '''

def mylist(val, lst=None): 
  lst = lst or [] 
  lst.append(val) 
  return lst

print mylist(1)
print mylist(2)
print mylist([1,2])
"""

"""
# Инверсия знаков в списке
# c сохранением исходного

a = [1, 2, 3, 4, 5]
print 'a (old)= ', a
b=[]
for i in range(len(a)):
    b.append(-a[i])
    i += 1

print  'a (new)= ', a
print  'b= ', b

# с изменением исходного списка

a = [1, 2, 3, 4, 5]
print  'a (old)= ', a

for i in range(len(a)):
    a[i]=-a[i]
    i += 1

print  'a (new)= ', a


# Работа с файлами

l = [str(i)+str(i-1) for i in range(20)]

f = open('text.txt', 'w')

for index in l:
    f.write(index + '\n')

f.close()

#f = open('text.txt', 'r')



import numpy as np

# на сколько синусоида длиннее прямой?
n = 1E3
# 
L0 = 0.0
# РїСЂР°РІР°СЏ РіСЂР°РЅРёС†Р°
L1 = 80 # cm

def integral(x):
    return np.sqrt(1 + np.power(np.sin((x )), 2))

if __name__ == '__main__':
    h = (L1 - L0) / n
    d1 = lambda r: (r - L1) *100 / L1

    result = h * sum([integral(xi - h / 2.0) for xi in np.arange(L0, L1, h)])
    print('<square> L(d) = {}'.format(result))
    print('delta = {} %'.format(d1(result)))

    result = 0
    x_j = lambda j: L0 + j * h
    for k in range(1, int(n), 2):
        result += integral(x_j(k-1)) + 4 * integral(x_j(k)) + integral(x_j(k+1))
    result *= h / 3.0
    print('<simpson> L(d) = {}'.format(result))
    print('delta = {} %'.format(d1(result)))


print sum(range(10))


def sum(lst, start): 
  return reduce(lambda x, y: x + y, lst, start) 

lst = range(10) 
f = lambda x, y: (x[0] + y, x[1]+[x[0] + y]) 
print reduce(f, lst, (0, []))


class Fibonacci: 
  ##Итератор последовательности Фибоначчи до N
 
  def __init__(self, N):  
    self.n, self.a, self.b, self.max = 0, 0, 1, N 
 
  def __iter__(self):  
    # сами себе итератор: в классе есть метод next()  
    return self 
 
  def next(self): 
    if self.n < self.max: 
      a, self.n, self.a, self.b = self.a, self.n+1, self.b, self.a+self.b 
      return a 
    else:  
      raise StopIteration 
 
# Использование:   
for i in Fibonacci(100): 
  print i, 
"""
# Простые генераторы
def Fib(N): 
  a, b = 0, 1 
  for i in xrange(N): 
    yield a 
    a, b = b, a + b 

for i in Fib(100): 
  print i,


























