from scipy.signal import convolve2d
import numpy as np

kernel = np.ones((3, 3), dtype=int)
matriz = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

resultado = convolve2d(matriz, kernel, mode='same', boundary='fill', fillvalue=0)
print(resultado)
