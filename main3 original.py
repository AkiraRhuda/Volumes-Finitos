import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Propriedades do Fluido (AR)
rho = 1.1769  # kg/m^3
mu = 1.85e-5  # Pa.s
Do2_ar = 2.015e-5  # m^2/s
r = 0.1  # kg/m^3.s

# Condições iniciais
Re = 100
Cw = 0.21  # Fração molar na parede

# Dimensões do domínio
Lx = 3  # m
Ly = 1  # m

# Discretização
nx = 60
ny = 20
N = nx * ny

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Velocidades
Ue = Re * mu / (rho * Ly)
Uw = Ue  # Assumindo simetria
Un = 0
Us = 0

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
        Wp = 2 * Do2_ar * (dy / dx + dx / dy)
        Ww = -Uw * dy / 2 - Do2_ar * dy / dx
        We = Ue * dy / 2 - Do2_ar * dy / dx
        Wn = Un * dx / 2 - Do2_ar * dx / dy
        Ws = -Us * dx / 2 - Do2_ar * dx / dy

        # Condições de contorno
        if i == 0:  # Fronteira oeste (entrada)
            Wp = Uw * dy + 2 * Do2_ar * dy / dx + Do2_ar * dx / dy
            B[m] = Uw * Cw * dy + Do2_ar * Cw * dy / dx
        elif i == nx - 1:  # Fronteira leste (saída)
            Wp = Ue * dy + Do2_ar * dy / dx + Do2_ar * dx / dy

        if j == 0:  # Fronteira norte
            Wp += Un * dx
        elif j == ny - 1:  # Fronteira sul
            Wp += -Us * dx

        # Termo fonte (apenas na região central do queimador)
        x_pos = i * dx
        y_pos = j * dy
        if (Lx / 3 - Lx / 20 <= x_pos <= Lx / 3 + Lx / 20) and (Ly / 2 - Ly / 20 <= y_pos <= Ly / 2 + Ly / 20):
            B[m] = -r * dx * dy / rho

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