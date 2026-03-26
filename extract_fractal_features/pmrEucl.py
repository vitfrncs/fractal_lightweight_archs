import numpy as np
from numba import njit

@njit
def pmrEucl(img, maxr):
    '''
    Calcula a matriz de probabilidades de uma imagem usano distância eclidiana 
    img deve ser uma ndarray numpy convertido em uint8 para float64
    maxr é o limite superior do rio, deve ser impar.
    '''
    aux = img.astype(np.float64)
    r = list(range(3, maxr + 1, 2))
    p = np.zeros((r[-1]**2, len(r)), dtype=np.float64)

    for k in range(len(r)):
        rk = r[k]
        ncaixas = (img.shape[0] - rk + 1) * (img.shape[1] - rk + 1)
        lim = (rk / 2) - 0.5

        for x in range(int(lim), img.shape[0] - int(lim)):
            for y in range(int(lim), img.shape[1] - int(lim)):
                m = 0
                xi = int( x - lim )
                xf = int( x + lim )
                yi = int( y - lim )
                yf = int( y + lim )

                for i in range(xi, xf + 1):
                    for j in range(yi, yf + 1):
                        dist = np.sqrt(
                            (aux[i, j, 0] - aux[x, y, 0])**2 + 
                            (aux[i, j, 1] - aux[x, y, 1])**2 + 
                            (aux[i, j, 2] - aux[x, y, 2])**2 
                        )
                        if dist <= rk:
                            m += 1
                p[m-1,k] += 1
        p[:, k] = p[:, k] / ncaixas

    return p