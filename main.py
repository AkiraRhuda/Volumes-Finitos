import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

########################################
############## PARÂMETROS ##############
########################################


# Propriedades do Fluido (AR)
rho = 1.1769  # kg/m^3
T = 300  # K
mu = 1.85*10**(-5) # Pa . s
Do2_ar = 2.015*10**(-5) # m^2 / s
r = 0.1 # kg / m . s

# Condições iniciais
Re = 100

# Parâmetros da Barra
Lx = 3  # m
Ly = 1  # m

# Termos da velocidade
Ue = Re * mu / (rho * Ly)
Uw = 0
Un = 0
Us = 0

Fe = rho*Ue
Fw = rho*Uw
Fn = rho*Un
Fs = rho*Us


# Parâmetros da simulação
nx = 60
ny = 120
N = nx * ny

dx = Lx/(nx - 1)
dy = Ly/(ny - 1)

A = np.zeros((N, N))
B = np.zeros(N)


########################################
############## EXPLÍCITO ###############
########################################

def Gauss_Seidel(A, b, x0, Eppara, maxit):
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)
    iter = 0
    Epest = np.linspace(100, 100, ne)

    while np.max(Epest) >= Eppara and iter <= maxit:
        x_old = np.copy(x)

        for i in range(ne):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Critério de parada
        Epest = np.abs((x - x_old) / x) * 100

        iter += 1

    return x



################ MONTAGEM DA MATRIZ ################
for i in range(1, ny - 1):
    for m in range(i * nx + 1, (i + 1) * nx - 1):
        Wp = Fe/2 - Fw/2 + Fn/2 - Fs/2 + 2*rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy
        Ww = -Fw/2 - rho*Do2_ar*dy/dx
        We = Fe/2 - rho*Do2_ar*dy/dx
        Ws = -Fs/2 - rho*Do2_ar*dx/dy
        Wn = Fn/2 - rho*Do2_ar*dx/dy
        S = r*dx*dy

        A[m, m] = Wp
        A[m, m - 1] = Ww
        A[m, m + 1] = We
        A[m, m + nx] = Wn
        A[m, m - nx] = Ws
        B[m] = S

# Canto Superior Esquerdo
m = 0
Ap = 1 + 4 * rxt + 4 * ryt
Ae = -4 / 3 * rxt
As = -4 / 3 * ryt
S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Tn * ryt

A[m, m] = Ap
A[m, m + 1] = Ae
A[m, m + nx] = As
B[m] = S

# Fronteira Norte
for m in range(1, nx - 1):
    Ap = 1 + 2 * rxt + 4 * ryt
    Aw = -rxt
    Ae = -rxt
    As = -4 / 3 * ryt
    S = Told[m] + 8 / 3 * Tn * ryt

    A[m, m] = Ap
    A[m, m - 1] = Aw
    A[m, m + 1] = Ae
    A[m, m + nx] = As
    B[m] = S

# Canto Superior Direito
m = nx - 1
Ap = 1 + 4 * rxt + 4 * ryt
Aw = -4 / 3 * rxt
As = -4 / 3 * ryt
S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Tn * ryt

A[m, m] = Ap
A[m, m - 1] = Aw
A[m, m + nx] = As
B[m] = S

# Fronteira Oeste
for m in range(nx, (ny - 2) * nx + 1, nx):
    Ap = 1 + 2 * rxt + 4 * ryt
    Ae = -4 / 3 * rxt
    An = -ryt
    As = -ryt
    S = Told[m] + 8 / 3 * Tw * rxt

    A[m, m] = Ap
    A[m, m + 1] = Ae
    A[m, m + nx] = As
    A[m, m - nx] = An
    B[m] = S

# Fronteira Leste
for m in range(2 * nx - 1, (ny - 1) * nx, nx):
    Ap = 1 + 2 * rxt + 4 * ryt
    Aw = -4 / 3 * rxt
    An = -ryt
    As = -ryt
    S = Told[m] + 8 / 3 * Te * rxt

    A[m, m] = Ap
    A[m, m - 1] = Aw
    A[m, m - nx] = An
    A[m, m + nx] = As
    B[m] = S

# Canto Inferior Esquerdo
m = (ny - 1) * nx
Ap = 1 + 4 * rxt + 4 * ryt
Ae = -4 / 3 * rxt
As = -4 / 3 * ryt
S = Told[m] + 8 / 3 * Tw * rxt + 8 / 3 * Ts * ryt

A[m, m] = Ap
A[m, m + 1] = Ae
A[m, m - nx] = An
B[m] = S

# Fronteira Sul
for m in range((ny - 1) * nx + 1, ny * nx - 1):
    Ap = 1 + 2 * rxt + 4 * ryt
    Aw = -rxt
    Ae = -rxt
    An = -4 / 3 * ryt
    S = Told[m] + 8 / 3 * Ts * ryt

    A[m, m] = Ap
    A[m, m - 1] = Aw
    A[m, m + 1] = Ae
    A[m, m - nx] = An
    B[m] = S

# Canto Inferior Direito
m = ny * nx - 1
Ap = 1 + 4 * rxt + 4 * ryt
Ae = -4 / 3 * rxt
As = -4 / 3 * ryt
S = Told[m] + 8 / 3 * Te * rxt + 8 / 3 * Ts * ryt

A[m, m] = Ap
A[m, m - 1] = Aw
A[m, m - nx] = An
B[m] = S


x0 = np.ones(N)
T = Gauss_Seidel(A, B, x0, Eppara, maxit)