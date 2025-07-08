import os 
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_pressure_layers_grid(P, wells, data_dir, max_layers=20):
    """
    Plota até 20 camadas do campo de pressão 3D em subplots (grade única).

    Parâmetros:
    - P: campo de pressão [NZ x NY x NX]
    - wells: lista de dicionários com chaves 'k', 'i', 'j', 'tipo', 'controle'
    - data_dir: diretório para salvar o arquivo PNG
    - max_layers: número máximo de camadas a serem plotadas
    """
    NZ = min(P.shape[0], max_layers)
    ncols = 5
    nrows = math.ceil(NZ / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for k in range(NZ):
        ax = axes[k]
        im = ax.imshow(P[k], cmap="jet", origin="lower")
        ax.set_title(f"Camada {k}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Marcar poços da camada k
        for w in wells:
            if w['k'] != k:
                continue
            i, j = w['i'], w['j']
            if w['tipo'] == 'INJETOR':
                ax.plot(j, i, 'o', color='violet', markersize=8, label='Injetor')
            elif w['tipo'] == 'PRODUTOR':
                ax.plot(j, i, 'v', color='magenta', markersize=8, label='Produtor')

        # Remover duplicatas na legenda local
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    # Remover eixos extras
    for idx in range(NZ, len(axes)):
        fig.delaxes(axes[idx])

    # Adiciona uma barra de cor geral
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="Pressão [kPa]")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    output_path = os.path.join(data_dir, f"mapa_pressao_subplots.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_permeability_layers_grid(kx, ky, wells, data_dir, max_layers=20):
    """
    Plota até max_layers camadas dos campos de permeabilidade kx e ky em subplots.

    Parâmetros:
    - kx, ky: arrays 3D de permeabilidade [NZ x NY x NX]
    - wells: lista de dicionários com 'i', 'j', 'k', 'tipo', 'controle'
    - data_dir: diretório onde as imagens serão salvas
    - max_layers: número máximo de camadas a plotar
    """
    NZ = min(kx.shape[0], max_layers)
    ncols = 5
    nrows = math.ceil(NZ / ncols)

    def get_marker_style(well):
        if well['tipo'] == 'INJETOR':
            return 'o', 'violet' if well['controle'] == 'VAZAO' else 'blue'
        elif well['tipo'] == 'PRODUTOR':
            return 'v', 'magenta' if well['controle'] == 'PRESSAO' else 'red'

    # --- kx ---
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes1 = axes1.flatten()

    for k in range(NZ):
        ax = axes1[k]
        im = ax.imshow(kx[k], cmap='viridis', origin='lower')
        ax.set_title(f"kx - Camada {k}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        for w in wells:
            i, j = w['i'], w['j']
            marker, color = get_marker_style(w)
            alpha = 1.0 if w['k'] == k else 0.2
            ax.plot(j, i, marker=marker, color=color, markersize=8, alpha=alpha)

    for idx in range(NZ, len(axes1)):
        fig1.delaxes(axes1[idx])

    cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="kx [mD]")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(data_dir, "mapa_kx_subplots.png"), dpi=300)
    plt.show()

    # --- ky ---
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes2 = axes2.flatten()

    for k in range(NZ):
        ax = axes2[k]
        im = ax.imshow(ky[k], cmap='viridis', origin='lower')
        ax.set_title(f"ky - Camada {k}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        for w in wells:
            i, j = w['i'], w['j']
            marker, color = get_marker_style(w)
            alpha = 1.0 if w['k'] == k else 0.2
            ax.plot(j, i, marker=marker, color=color, markersize=8, alpha=alpha)

    for idx in range(NZ, len(axes2)):
        fig2.delaxes(axes2[idx])

    cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="ky [mD]")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(data_dir, "mapa_ky_subplots.png"), dpi=300)
    plt.show()

