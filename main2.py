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
r = 0 # kg / m . s

# Condições iniciais
Re = 100
Cw = 0.21

# Parâmetros da Barra
Lx = 3  # m
Ly = 1  # m

# Termos da velocidade
Ue = Re * mu / (rho * Ly)
Uw = Re * mu / (rho * Ly)
Un = 0
Us = 0

Fe = rho*Ue
Fw = rho*Uw
Fn = rho*Un
Fs = rho*Us

# Parâmetros da simulação

nx = 2
ny = 2
N = nx * ny

dx = Lx/(nx - 1)
dy = Ly/(ny - 1)

A = np.zeros((N, N))
B = np.zeros(N)

x = np.linspace(dx/2, Lx - dx/2, nx)
y = np.linspace(dx/2, Ly - dx/2, ny)
X, Y = np.meshgrid(x, y)


########################################
############## IMPLÍCITO ###############
########################################

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

index = np.arange(N).reshape(ny,nx)

# Central
for i in range(1,ny-1):
  for j in range(i*nx+1,(i+1)*nx-1):
    A[j,j] = Fe/2 - Fw/2 + Fn/2 - Fs/2 + 2*rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy # Ap
    A[j,j-1] = -Fw/2 - rho*Do2_ar*dy/dx # Aw
    A[j,j+1] = Fe/2 - rho*Do2_ar*dy/dx # Ae
    A[j,j-nx] = Fn/2 - rho*Do2_ar*dx/dy # An
    A[j,j+nx] = -Fs/2 - rho*Do2_ar*dx/dy # As
    B[j] = r*dx*dy # Fonte

# Fronteira Norte
for i in range(1,nx-1):
  A[i,i] = Fe/2 - Fw/2 - Fs/2 + 2*rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy # Ap
  A[i,i-1] = -Fw/2 - rho*Do2_ar*dy/dx # Aw
  A[i,i+1] = Fe/2 - rho*Do2_ar*dy/dx # Ae
  A[i,i+nx] = -Fs/2 - rho*Do2_ar*dx/dy # As
  B[i] = r*dx*dy # Fonte

# Fronteira Sul
for i in range((ny-1)*nx+1,ny*nx-1):
  A[i,i] = Fe/2 - Fw/2 + Fn/2 + 2*rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy # Ap
  A[i,i-1] = -Fw/2 - rho*Do2_ar*dy/dx # Aw
  A[i,i+1] = Fe/2 - rho*Do2_ar*dy/dx # Ae
  A[i,i-nx] = Fn/2 - rho*Do2_ar*dx/dy # An
  B[i] = r*dx*dy # Fonte

# Fronteira Leste
for i in range(2*nx-1,(ny-1)*nx,nx):
  A[i,i] = -Fw/2 + Fn/2 - Fs/2 + rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy # Ap
  A[i,i-1] = -Fw/2 - rho*Do2_ar*dy/dx # Aw
  A[i,i-nx] = Fn/2 - rho*Do2_ar*dx/dy # An
  A[i,i+nx] = -Fs/2 - rho*Do2_ar*dx/dy # As
  B[i] = r*dx*dy # Fonte

# Fronteira Oeste
for i in range(nx,(ny-2)*nx+1,nx):
  A[i,i] = Fe/2 + Fn/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx/2) + rho*Do2_ar*dx/dy + rho*Do2_ar*dx/dy # Ap
  A[i,i+1] = Fe/2 - rho*Do2_ar*dy/dx # Ae
  A[i,i-nx] = Fn/2 - rho*Do2_ar*dy/dx # An
  A[i,i+nx] = -Fs/2 - rho*Do2_ar*dx/dy # As
  B[i] = r*dx*dy + Fw*Cw + rho*Do2_ar*dy/(dx/2)*Cw # Fonte

# Canto Superior Direito
i = nx-1
A[i,i] = -Fw/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy # Ap
A[i,i-1] = -Fw/2 - rho*Do2_ar*dy/dx # Aw
A[i,i+nx] = -Fs/2 - rho*Do2_ar*dx/dy # As
B[i] = r*dx*dy # Fonte

# Canto Superior Esquerdo
i = 0
A[i,i] = Fe/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx/2) + rho*Do2_ar*dx/dy # Ap
A[i,i+1] = Fe/2 - rho*Do2_ar*dy/dx # Ae
A[i,i+nx] = -Fs/2 - rho*Do2_ar*dx/dy # As
B[i] = r*dx*dy + Fw*Cw + rho*Do2_ar*dy/(dx/2)*Cw # Fonte

# Canto Inferior Direito
i = ny*nx-1
A[i,i] = -Fw/2 + Fn/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy # Ap
A[i,i-1] = -Fw/2 - rho*Do2_ar*dy/dx # Aw
A[i,i-nx] = Fn/2 - rho*Do2_ar*dx/dy # An
B[i] = r*dx*dy # Fonte

# Canto Inferior Esquerdo
i = (ny-1)*nx
A[i,i] = Fe/2 + Fn/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx/2) + rho*Do2_ar*dx/dy # Ap
A[i,i+1] = Fe/2 - rho*Do2_ar*dy/dx # Ae
A[i,i-nx] = Fn/2 - rho*Do2_ar*dy/dx # An
B[i] = r*dx*dy + Fw*Cw + rho*Do2_ar*dy/(dx/2)*Cw # Fonte

# Solução do sistema Linear
x0 = np.ones(N)
T = Gauss_Seidel(A, B, x0, scarvorought(12), maxit=1000)

T_new = np.zeros((ny, nx))

# Armazenando a temperatura atual
for i in range(ny):
    T_new[(ny-1)-i, :] = T[index[i, :]]



fig, ax = plt.subplots(figsize=(13, 7))
c = ax.contourf(X, Y, T_new, cmap='plasma')
fig.colorbar(c, ax=ax, label='Temperatura (°C)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
plt.title('Distrubuição da Solução Implícita da Equação de Calor 2D')
plt.show()

fig2 = plt.figure(figsize=(13, 13))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X, Y, T_new, cmap='viridis')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('T(x,y) (°C)')
ax2.view_init(elev=30, azim=55)
plt.title('Solução Implícita da Equação de Calor 2D')
fig2.colorbar(surf2, shrink=0.5, aspect=10)
ax2.view_init(azim=245)
plt.show()