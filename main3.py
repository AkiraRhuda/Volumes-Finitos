import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

########################################
############## PARÂMETROS ##############
########################################


# Propriedades do Fluido (AR)
rho = 1.1769  # kg/m^3
T = 300  # K
mu = 1.85*10**(-5) # Pa . s
Do2_ar = 2.015*10**(-5) # m^2 / s
r = 0.1 # kg / m^3 . s

# Condições iniciais
Re = 100
Cw =  21

# Parâmetros da Barra
Lx = 3  # m
Ly = 1  # m

# Parâmetros da simulação
nx = 120
ny = 40
N = nx * ny

dx = Lx/(nx - 1)
dy = Ly/(ny - 1)

x = np.linspace(dx/2, Lx - dx/2, nx)
y = np.linspace(dx/2, Ly - dx/2, ny)
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

# Construção da matriz esparsa (forma mais eficiente)
from scipy.sparse import lil_matrix

A = lil_matrix((N, N))
B = np.zeros(N)


# Função para converter índices 2D para 1D
def idx(i, j):
    return i * ny + j


# Montagem da matriz
for i in range(nx):
    for j in range(ny):
        m = idx(i, j)

        # Termos gerais
        Wp = Fe/2 - Fw/2 + Fn/2 - Fs/2 + 2*rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy
        Ww = -Fw/2 - rho*Do2_ar*dy/dx
        We = Fe/2 - rho*Do2_ar*dy/dx
        Wn = Fn/2 - rho*Do2_ar*dx/dy
        Ws = -Fs/2 - rho*Do2_ar*dx/dy

        # Condições de contorno
        if i == 0:  # Fronteira oeste (entrada)
            Wp = Fe/2 + Fn/2 - Fs/2 + rho*Do2_ar*dy/dx + rho*Do2_ar*dy/(dx/2)
            B[m] = Fw*Cw + rho*Do2_ar*dy/(dx/2)*Cw
        elif i == nx - 1:  # Fronteira leste (saída)
            Wp = Fe - Fw/2 + Fn/2 - Fs/2 + rho*Do2_ar*dy/dx + 2*rho*Do2_ar*dx/dy

        if j == 0:  # Fronteira norte
            Wp = Fe / 2 - Fw / 2 - Fs / 2 + 2 * rho * Do2_ar * dy / dx + rho * Do2_ar * dx / dy
        elif j == ny - 1:  # Fronteira sul
            Wp = Fe/2 - Fw/2 + Fn/2 + 2*rho*Do2_ar*dy/dx + rho*Do2_ar*dx/dy

        # Termo fonte (apenas na região central do queimador)
        x_pos = i * dx
        y_pos = j * dy
        if (Lx / 3 - Lx / 20 <= x_pos <= Lx / 3 + Lx / 20) and (Ly / 2 - Ly / 20 <= y_pos <= Ly / 2 + Ly / 20):
            B[m] = -r*dx*dy

        # Preencher a matriz
        A[m, m] = Wp
        if i > 0:
            A[m, idx(i - 1, j)] = Ww
        if i < nx - 1:
            A[m, idx(i + 1, j)] = We
        if j > 0:
            A[m, idx(i, j - 1)] = Wn
        if j < ny - 1:
            A[m, idx(i, j + 1)] = Ws

# Resolver o sistema
from scipy.sparse.linalg import spsolve

A_csr = A.tocsr()
W = spsolve(A_csr, B)

# Reorganizar os resultados
XYW = W.reshape((nx, ny)).T

# Visualização
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Plot 2D
plt.figure(figsize=(13, 7))
contour = plt.contourf(X, Y, XYW, levels=50, cmap='plasma')
plt.colorbar(contour, label='Concentração de O₂')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Distribuição da Espécie Química O₂')
plt.show()

# Plot 3D
fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, XYW, cmap='plasma', edgecolor='none')
fig.colorbar(surf, ax=ax, label='Concentração de O₂')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Concentração')
ax.set_title('Distribuição 3D da Espécie Química O₂')
plt.show()