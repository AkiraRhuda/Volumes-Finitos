import math
import numpy as np
#import pandas as pd
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
r = 0 # kg / m^3 . s

# Condições iniciais
Re = 100
Cw = 21

# Parâmetros da Barra
Lx = 3  # m
Ly = 1  # m

# Parâmetros da simulação
nx = 120
ny = 40
N = nx * ny

dx = Lx/(nx - 1)
dy = Ly/(ny - 1)

A = np.zeros((N, N))
B = np.zeros(N)

x = np.linspace(0, Lx , nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Termos da velocidade
Ue = Re * mu / (rho * Ly)
Uw = Re * mu / (rho * Ly)
Un = 0
Us = 0

Fe = rho*Ue*dy
Fw = rho*Uw*dy
Fn = rho*Un*dx
Fs = rho*Us*dx

########################################
############## EXPLÍCITO ###############
########################################

def Gauss_Seidel_com_relaxamento(A, b, Lambda, x0, Eppara, maxit):
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)

    iter = 0
    Epest = np.linspace(100,100,ne)

    while np.max(Epest) >= Eppara and iter <= maxit:
        x_old = np.copy(x)

        for i in range(ne):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Critério de parada
        Epest = np.abs((x - x_old) / x) * 100

        iter += 1

        # Relaxamento
        x = Lambda*x + (1-Lambda)*x_old

    return x

def Gauss_Seidel(A, b, x0, Eppara, maxit):
    ne = len(b)
    x = np.zeros(ne) if x0 is None else np.array(x0)
    iter = 0
    Epest = np.linspace(100,100,ne)

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

def scarvorought(n):
    return 0.5 * 10 ** (2-n)



################ MONTAGEM DA MATRIZ ################

# Central
for i in range(1, ny - 1):
    for m in range(i * nx + 1, (i + 1) * nx - 1):
        Wp = Fe/2 - Fw/2 + Fn/2 - Fs/2 + 2*rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy
        Ww = -Fw/2 - rho*Do2_ar*dy/dx
        We = Fe/2 - rho*Do2_ar*dy/dx
        Ws = -Fs/2 - rho*Do2_ar*dx/dy
        Wn = Fn/2 - rho*Do2_ar*dx/dy
        S = 0

        A[m, m] = Wp
        A[m, m - 1] = Ww
        A[m, m + 1] = We
        A[m, m + nx] = Wn
        A[m, m - nx] = Ws
        B[m] = S

# Canto Inferior Esquerdo
m = 0
Wp = Fe/2 + Fn/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx) + rho*Do2_ar*dx/dy
We = Fe/2 - rho*Do2_ar*dy/dx
Wn = Fn/2 - rho*Do2_ar*dx/dy
S = Fw*Cw + rho*Do2_ar*dy/(dx)*Cw

A[m, m] = Wp
A[m, m + 1] = We
A[m, m + nx] = Wn
B[m] = S



# Fronteira Sul
for m in range(1, nx - 1):
    Wp = Fe/2 - Fw/2 + Fn/2 + 2*rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy
    Ww = -Fw/2 - rho*Do2_ar*dy/dx
    We = Fe/2 - rho*Do2_ar*dy/dx
    Wn = Fn/2 - rho*Do2_ar*dx/dy
    S = 0

    A[m, m] = Wp
    A[m, m - 1] = Ww
    A[m, m + 1] = We
    A[m, m + nx] = Wn
    B[m] = S

# Canto Inferior Direito
m = nx - 1
Wp = Fe - Fw/2 + Fn/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy
Ww = -Fw/2 - rho*Do2_ar*dy/dx
Wn = Fn/2 - rho*Do2_ar*dx/dy
S = 0

A[m, m] = Wp
A[m, m - 1] = Ww
A[m, m + nx] = Wn
B[m] = S

# Fronteira Oeste
for m in range(nx, (ny - 2) * nx + 1, nx):
    Wp = Fe/2 + Fn/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx) + rho*Do2_ar*dx/dy + rho*Do2_ar*dx/dy
    We = Fe/2 - rho*Do2_ar*dy/dx
    Wn = Fn/2 - rho*Do2_ar*dx/dy
    Ws = -Fs/2 - rho*Do2_ar*dx/dy
    S = Fw*Cw + rho*Do2_ar*dy/(dx)*Cw

    A[m, m] = Wp
    A[m, m + 1] = We
    A[m, m - nx] = Ws
    A[m, m + nx] = Wn
    B[m] = S

# Fronteira Leste
for m in range(2 * nx - 1, (ny - 1) * nx, nx):
    Wp = Fe - Fw/2 + Fn/2 - Fs/2 + rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy
    Ww = -Fw/2 - rho*Do2_ar*dy/dx
    Wn = Fn/2 - rho*Do2_ar*dx/dy
    Ws = -Fs/2 - rho*Do2_ar*dx/dy
    S = 0

    A[m, m] = Wp
    A[m, m - 1] = Ww
    A[m, m + nx] = Wn
    A[m, m - nx] = Ws
    B[m] = S

# Canto Superior Esquerdo
m = (ny - 1) * nx
Wp = Fe/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx) + rho*Do2_ar*dx/dy
We = Fe/2 - rho*Do2_ar*dy/dx
Ws = -Fs/2 - rho*Do2_ar*dx/dy
S = Fw*Cw + rho*Do2_ar*dy/(dx)*Cw

A[m, m] = Wp
A[m, m + 1] = We
A[m, m - nx] = Ws
B[m] = S

# Fronteira Norte
for m in range((ny - 1) * nx + 1, ny * nx - 1):
    Wp = Fe / 2 - Fw / 2 - Fs / 2 + 2 * rho * Do2_ar * dy / dx + rho * Do2_ar * dx / dy
    Ww = -Fw / 2 - rho * Do2_ar * dy / dx
    We = Fe / 2 - rho * Do2_ar * dy / dx
    Ws = -Fs / 2 - rho * Do2_ar * dx / dy
    S = 0

    A[m, m] = Wp
    A[m, m - 1] = Ww
    A[m, m + 1] = We
    A[m, m - nx] = Ws
    B[m] = S

# Canto Superior Direito
m = ny * nx - 1
Wp = Fe - Fw/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy
Ww = -Fw/2 - rho*Do2_ar*dy/dx
Ws = -Fs/2 - rho*Do2_ar*dx/dy
S = 0

A[m, m] = Wp
A[m, m - 1] = Ww
A[m, m - nx] = Ws
B[m] = S

# Termo fonte (apenas na região central do queimador)
for i in range(nx):
    for m in range(ny):
        x_pos = i * dx
        y_pos = m * dy
        if (Lx / 2 - Lx / 20 <= x_pos <= Lx / 2 + Lx / 20) and (Lx / 2 - Lx / 20 <= y_pos <= Lx / 2 + Lx / 20):
            B[m] = -r * dx * dy / rho


x0 = np.ones(N)
Eppara = scarvorought(12)
Lambda = 1
W = np.linalg.solve(A, B)
index = np.arange(N).reshape(ny,nx)
XYW = np.zeros((ny, nx))
for i in range(ny):
        XYW[(ny-1)-i, :] = W[index[i, :]]

fig, ax = plt.subplots(figsize=(13, 7))
c = ax.contourf(X, Y, XYW, cmap='plasma')
fig.colorbar(c, ax=ax, label=r'Concentração ')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
plt.title(r'Distrubuição da Espécie química $O_2$')
plt.show()
print('A')