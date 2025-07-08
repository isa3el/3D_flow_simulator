import numpy as np
import os
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from simulation_functions import read_input_file, update_properties, build_simulator_3D, compute_well_flows_3D, export_well_results_3D
from plot_functions import plot_pressure_layers_grid, plot_permeability_layers_grid, plot_all_pressure_maps_with_values

# Para rodar o simulador, defina o diretório que contém os arquivos .txt de entrada.
# Descomente os diretório que será utilizado.


# Diretórios disponíveis:
# - Grid_60_60_3: reservatório com malha 60x60x3 e 9 poços (5 produtores controlados por BHP e 4 injetores controlados por vazão).
# - Grid_15_15_5: reservatório com malha 15x15x5 e 3 poços (2 produtores controlados por BHP e 1 injetor controlado por vazão).
# - Grid_5_5_5: reservatório com malha 5x5x5 e 2 poços (1 produtor controlado por BHP e 1 injetor controlado por vazão).

# Os diretórios disponíveis possuem "60_60_3", "15_15_5" e "5_5_5" no nome apenas para fins de identificação.
# O simulador é genérico e capaz de rodar reservatórios de qualquer tamanho, desde que os arquivos de entrada estejam no formato esperado.
# Disponibilizei o código no meu github. É possivel acessar em https://github.com/isa3el/3D_flow_simulator
# Descomente abaixo o diretório que será utilizado:

data_dir = "Grid_60_60_3"
#data_dir = "Grid_15_15_5"
#data_dir = "Grid_5_5_5"

#Definindo os arquivos a serem lidos a partir da 
input_path = os.path.join(data_dir, "input.txt")
grid_path = os.path.join(data_dir, "grid.txt")
pres_path = os.path.join(data_dir, "pres.txt")
poro_path = os.path.join(data_dir, "poro.txt")
permx_path = os.path.join(data_dir, "perm_x.txt")
permy_path = os.path.join(data_dir, "perm_y.txt")
permz_path = os.path.join(data_dir, "perm_z.txt")

NX, NY, NZ, dx, dy, dz, h, active, kx, ky, kz, phi_ref, press_init, mu_ref, rho_ref, wells, p_ref, c_r, c_f, c_mu = read_input_file(input_path, grid_path, permx_path, permy_path, permz_path, pres_path, poro_path)

mu_p, rho_p, phi_p = update_properties(press_init, p_ref, mu_ref, rho_ref, phi_ref, c_mu, c_f, c_r)

# Inicialização de parâmetros do tempo e gravidade
dt = 1.0  # passo de tempo 
n_steps = 5  # número de passos de tempo
g = 9.81  # gravidade

# Coordenada z de cada célula (para efeito de gravidade)
z_coords = np.zeros((NZ, NY, NX))
for k in range(NZ):
    z_coords[k, :, :] = h * k  # camadas horizontais e planas com espessura h


P = build_simulator_3D(
    NX, NY, NZ, dx, dy, dz, h,
    active, kx, ky, kz,
    press_init, phi_ref,
    mu_ref, rho_ref,
    p_ref, c_mu, c_f, c_r,
    wells, n_steps, dt, g, z_coords
)
#print(P)

wells_prod_wi = compute_well_flows_3D(P, kx, ky, dz, dx, dy, phi_ref, wells, mu_ref, rho_ref, p_ref, c_mu, c_f, c_r)

plot_pressure_layers_grid(P, wells, data_dir)
plot_permeability_layers_grid(kx, ky, wells, data_dir)

export_well_results_3D(wells_prod_wi, data_dir, NX, NY, NZ, dx, dy, dz, h, mu_ref, rho_ref, p_ref, c_mu, c_r, c_f, P, kx, ky, kz, phi_ref)