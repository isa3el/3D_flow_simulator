import numpy as np
import os
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def read_input_file(filename, grid_file, permx_file, permy_file, permz_file, press_file, poro_file):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    def extract_block(tag):
        try:
            start = lines.index(f'[{tag}]') + 1
            end = next((i for i in range(start, len(lines)) if lines[i].startswith('[')), len(lines))
            return lines[start:end]
        except ValueError:
            raise ValueError(f"Tag [{tag}] não encontrada no arquivo.")

    # Leitura do grid XY
    NX, NY, NZ = map(int, extract_block("NX NY NZ")[0].split())
    dx_block = extract_block("DX")[0].split('*')
    dx = [float(dx_block[1])] * int(dx_block[0]) if len(dx_block) == 2 else list(map(float, dx_block))
    dy_block = extract_block("DY")[0].split('*')
    dy = [float(dy_block[1])] * int(dy_block[0]) if len(dy_block) == 2 else list(map(float, dy_block))
    dz_block = extract_block("DZ")[0].split('*')
    dz = [float(dz_block[1])] * int(dz_block[0]) if len(dy_block) == 2 else list(map(float, dz_block))
    h = float(extract_block("H")[0])

    # Leitura de arquivos externos
    active = np.loadtxt(grid_file).reshape((NZ, NY, NX))
    kx = np.loadtxt(permx_file).reshape((NZ, NY, NX))
    ky = np.loadtxt(permy_file).reshape((NZ, NY, NX))
    kz = np.loadtxt(permz_file).reshape((NZ, NY, NX))
    phi_ref = np.loadtxt(poro_file).reshape((NZ, NY, NX))
    press_init = np.loadtxt(press_file).reshape((NZ, NY, NX))
    
    # Poços
    wells_block = extract_block("WELLS")
    n_wells = int(wells_block[0])
    wells = []
    for line in wells_block[1:n_wells + 1]:
        tokens = line.split()
        if len(tokens) == 7:
            tipo_raw, rw, i, j, k, controle, valor = tokens
        else:
            raise ValueError(f"Linha inválida no bloco [WELLS]: {line}")

        tipo = "INJETOR" if tipo_raw.upper() == "INJ" else "PRODUTOR"
        well = {
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "rw": float(rw),
            "tipo": tipo,
            "controle": controle.upper(),
            "valor": float(valor)
        }
        wells.append(well)

    def get_scalar(tag):
        block = extract_block(tag)
        if block:
            return float(block[0])
        else:
            warnings.warn(f"[{tag}] não encontrado.")
            return None

    def get_vector(tag, default_nz=1, default_val=1.0):
        block = extract_block(tag)
        if block:
            val = block[0].split('*')
            if len(val) == 2:
                return [float(val[1])] * int(val[0])
            else:
                return list(map(float, block[0].split()))
        else:
            warnings.warn(f"[{tag}] não encontrado. Usando vetor constante.")
            return [default_val] * default_nz

    # Propriedades dependentes de pressão
    mu_ref = get_scalar("VISREF")
    rho_ref = get_scalar("RHOREF")
    p_ref = get_scalar("PREF")
    c_mu   = get_scalar("VISPR")
    c_f    = get_scalar("COMPF")
    c_r    = get_scalar("COMPR")

    # verificação
    print("=== MALHA ===")
    print("NX, NY, NZ:", NX, NY, NZ)
    print("dx:", dx)
    print("dy:", dy)
    print("dz:", dz)
    print("h (espessura célula):", h)
    print("\n=== PROPRIEDADES ===")
    print(f"mu_ref: {mu_ref} cP | rho_ref: {rho_ref} kg/m³")
    print(f"c_mu: {c_mu} | c_r: {c_r} | c_f: {c_f} | p_ref: {p_ref}")
    print("\n=== POÇOS ===")
    for w in wells:
        print(w)

    return NX, NY, NZ, dx, dy, dz, h, active, kx, ky, kz, phi_ref, press_init, mu_ref, rho_ref, wells, p_ref, c_r, c_f, c_mu


def update_properties(p, p_ref, mu_ref, rho_ref, phi_ref, c_mu, c_f, c_r):
    """
    Atualiza propriedades físicas do fluido e da rocha com base na pressão local.

    Parâmetros:
    - p: pressão [kPa] (escalar ou np.ndarray)
    - p_ref: pressão de referência [kPa]
    - mu_ref: viscosidade de referência [cP]
    - rho_ref: densidade de referência [kg/m³]
    - phi_ref: porosidade de referência (escalar ou ndarray)
    - c_mu: variação da viscosidade com pressão [cP/kPa]
    - c_f: compressibilidade do fluido [1/kPa]
    - c_r: compressibilidade da rocha [1/kPa]

    Retorno:
    - mu_p: viscosidade [cP]
    - rho_p: densidade [kg/m³]
    - phi_p: porosidade
    """

    delta_p = p - p_ref

    mu_p = mu_ref + c_mu * delta_p
    rho_p = rho_ref * (1 + c_f * delta_p)
    phi_p = phi_ref * (1 + c_r * delta_p)

    return mu_p, rho_p, phi_p



def build_simulator_3D(NX, NY, NZ, dx, dy, dz, h, active, kx, ky, kz,
                                              press_init, phi_ref, mu_ref, rho_ref,
                                              p_ref, c_mu, c_f, c_r, wells):
    """
    Simulador de escoamento monofásico em malha 3D com propriedades dependentes da pressão
    e acoplamento completo entre as direções X, Y e Z (vertical).

    Este simulador resolve, para cada camada k, o sistema de equações de fluxo considerando:

    - Propriedades (viscosidade, densidade e porosidade) atualizadas ponto a ponto com base na pressão local, conforme relações lineares dependentes da pressão.

- Transmissibilidades entre células vizinhas nas direções X, Y e Z, considerando harmônicas das permeabilidades e médias das viscosidades.

- Poços posicionados em células específicas (i, j, k), com controle por vazão ou pressão, incluindo cálculo do índice de produtividade (WI) com viscosidade local.

- Resolução do sistema linear completo T·P = Q usando matriz esparsa CSR (SciPy).

Retorna:
- P: matriz 3D (NZ x NY x NX) com o campo de pressão final em kPa.
"""
    def get_index(i, j, k):
        return k * NY * NX + i * NX + j

    N = NX * NY * NZ
    T = sp.lil_matrix((N, N))
    Q = np.zeros(N)

    for k in range(NZ):
        for i in range(NY):
            for j in range(NX):
                if active[k, i, j] == 0:
                    continue

                idx = get_index(i, j, k)
                p_ijk = press_init[k, i, j]
                phi_ijk = phi_ref[k, i, j]

                mu_ijk, _, _ = update_properties(p_ijk, p_ref, mu_ref, rho_ref, phi_ijk, c_mu, c_f, c_r)

                # Transmissibilidade X (j±1)
                for dj in [-1, 1]:
                    nj = j + dj
                    if 0 <= nj < NX and active[k, i, nj]:
                        idx_n = get_index(i, nj, k)
                        mu_n, _, _ = update_properties(press_init[k, i, nj], p_ref, mu_ref, rho_ref,
                                                       phi_ref[k, i, nj], c_mu, c_f, c_r)
                        mu_eff = 0.5 * (mu_ijk + mu_n)
                        kx_eff = 2 * kx[k, i, j] * kx[k, i, nj] / (kx[k, i, j] + kx[k, i, nj])
                        Tx = (kx_eff * dz[k]) / (mu_eff * (dx[j] + dx[nj]) / 2)
                        T[idx, idx] += Tx
                        T[idx, idx_n] -= Tx

                # Transmissibilidade Y (i±1)
                for di in [-1, 1]:
                    ni = i + di
                    if 0 <= ni < NY and active[k, ni, j]:
                        idx_n = get_index(ni, j, k)
                        mu_n, _, _ = update_properties(press_init[k, ni, j], p_ref, mu_ref, rho_ref,
                                                       phi_ref[k, ni, j], c_mu, c_f, c_r)
                        mu_eff = 0.5 * (mu_ijk + mu_n)
                        ky_eff = 2 * ky[k, i, j] * ky[k, ni, j] / (ky[k, i, j] + ky[k, ni, j])
                        Ty = (ky_eff * dz[k]) / (mu_eff * (dy[i] + dy[ni]) / 2)
                        T[idx, idx] += Ty
                        T[idx, idx_n] -= Ty

                # Transmissibilidade Z (k±1)
                for dk in [-1, 1]:
                    nk = k + dk
                    if 0 <= nk < NZ and active[nk, i, j]:
                        idx_n = get_index(i, j, nk)
                        mu_n, _, _ = update_properties(press_init[nk, i, j], p_ref, mu_ref, rho_ref,
                                                       phi_ref[nk, i, j], c_mu, c_f, c_r)
                        mu_eff = 0.5 * (mu_ijk + mu_n)
                        kz_eff = 2 * kz[k, i, j] * kz[nk, i, j] / (kz[k, i, j] + kz[nk, i, j])
                        Tz = (kz_eff * dx[j] * dy[i]) / (mu_eff * (dz[k] + dz[nk]) / 2)
                        T[idx, idx] += Tz
                        T[idx, idx_n] -= Tz

    # Poços
    for w in wells:
        i, j, k = w["i"], w["j"], w["k"]
        if active[k, i, j] == 0:
            continue
        idx = get_index(i, j, k)
        ctrl = w["controle"]
        val = w["valor"]
        rw = w.get("rw", 0.1)

        p_w = press_init[k, i, j]
        mu_w, _, _ = update_properties(p_w, p_ref, mu_ref, rho_ref, phi_ref[k, i, j], c_mu, c_f, c_r)

        if ctrl == "VAZAO":
            Q[idx] += val
        elif ctrl == "PRESSAO":
            kx_ij = kx[k, i, j]
            ky_ij = ky[k, i, j]
            dx_ = dx[j]
            dy_ = dy[i]

            term1 = np.sqrt(ky_ij / kx_ij) * dx_**2
            term2 = np.sqrt(kx_ij / ky_ij) * dy_**2
            numerator = np.sqrt(term1 + term2)
            denominator = (ky_ij / kx_ij)**0.25 + (kx_ij / ky_ij)**0.25
            req = 0.28 * numerator / denominator
            log_term = np.log(req / rw)
            log_term = max(log_term, 1e-6)

            WI = (2 * np.pi * np.sqrt(kx_ij * ky_ij) * dz[k]) / log_term
            T_poço = WI / mu_w
            T[idx, idx] += T_poço
            Q[idx] += T_poço * val

    # Resolver sistema linear global
    T = T.tocsr()
    P_flat = spla.spsolve(T, Q)
    P_result = P_flat.reshape((NZ, NY, NX))

    return P_result

def compute_well_flows_3D(P, kx, ky, dz, dx, dy, poro, wells,
                          mu_ref, rho_ref, p_ref, c_mu, c_f, c_r):
    """
    Calcula a vazão dos poços levando em conta a dependência das propriedades físicas (viscosidade, densidade e porosidade) com a pressão local.

    - Se o poço for controlado por PRESSÃO:
        - Aplica a equação do índice de produtividade (WI), considerando anisotropia (kx, ky), raio do poço (rw) e espessura da camada (dz[k]).
        - Atualiza a viscosidade local com base na pressão do reservatório (P[k, i, j]) e calcula a vazão como: q = WI * (pressao_do_poco - pressao_no_reservatorio) / mu(p)

    - Se o poço for controlado por VAZÃO:
        - A vazão é definida diretamente pelo valor informado.
        - A pressão é lida diretamente da célula onde o poço está localizado (P[k, i, j]).

    Parâmetros:
    - P: campo de pressão 3D [NZ x NY x NX] (em kPa)
    - kx, ky: campos de permeabilidade nas direções X e Y [NZ x NY x NX] (em mD)
    - dz: vetor com espessura de cada camada [m]
    - dx, dy: vetores com tamanhos das células nas direções X e Y [m]
    - poro: campo de porosidade [NZ x NY x NX]
    - wells: lista de dicionários, cada um representando um poço com:
        - i, j: posição (linha, coluna)
        - tipo: 'INJETOR' ou 'PRODUTOR'
        - controle: 'VAZAO' ou 'PRESSAO'
        - valor: pressão alvo [kPa] ou vazão [m³/dia]
        - rw: raio do poço [m] (opcional, default = 0.1 m)

    - mu_ref, rho_ref: viscosidade e densidade de referência
    - p_ref: pressão de referência [kPa]
    - c_mu, c_f, c_r: coeficientes de variação com pressão para viscosidade, densidade e porosidade

    Retorna:
    - Lista de tuplas com os dados dos poços por camada:
        (k, i, j, tipo, controle, pressao [kPa], vazao [m³/dia])
"""

    results = []

    for w in wells:
        i, j, k = w['i'], w['j'], w['k']
        ctrl = w['controle']
        tipo = w['tipo']
        rw = w.get('rw', 0.1)
        val = w['valor']

        
        p_res = P[k, i, j]

        mu_p, _, _ = update_properties(p_res, p_ref, mu_ref, rho_ref, poro[k, i, j], c_mu, c_f, c_r)

        if ctrl == "PRESSAO":
            p_well = val

            kx_ij = kx[k, i, j]
            ky_ij = ky[k, i, j]
            dx_ = dx[j]
            dy_ = dy[i]

            term1 = np.sqrt(ky_ij / kx_ij) * dx_**2
            term2 = np.sqrt(kx_ij / ky_ij) * dy_**2
            numerator = np.sqrt(term1 + term2)
            denominator = (ky_ij / kx_ij)**0.25 + (kx_ij / ky_ij)**0.25
            req = 0.28 * numerator / denominator

            WI = (2 * np.pi * np.sqrt(kx_ij * ky_ij) * dz[k]) / (np.log(req / rw) + 1e-6)

            q = WI * (p_well - p_res) / mu_p

            results.append((k, i, j, tipo, ctrl, p_well, q))

        elif ctrl == "VAZAO":
            q = val
            results.append((k, i, j, tipo, ctrl, p_res, q))

    return results

def export_well_results_3D(well_results, data_dir,
                           NX, NY, NZ, dx, dy, dz,
                           h, mu_ref, rho_ref,
                           p_ref, c_mu, c_r, c_f,
                           P, kx, ky, kz, poro):
    """
    Exporta os resultados dos poços simulados em 3D com propriedades dependentes da pressão.

    Parâmetros:
    - well_results: lista de tuplas (k, i, j, tipo, controle, pressão, vazão)
    - data_dir: diretório onde salvar o arquivo de saída
    - NX, NY, NZ: dimensões da malha
    - dx, dy, dz: tamanhos das células por direção
    - h: altura nominal original usada como fallback [m]
    - mu_ref, rho_ref: propriedades de referência
    - p_ref: pressão de referência [kPa]
    - c_mu, c_r, c_f: coeficientes de variação com pressão
    - P, kx, ky, kz, poro: campos 3D (NZ x NY x NX)
    """

    output_path = os.path.join(data_dir, "results_3D.txt")

    with open(output_path, 'w') as f:
        f.write("=== INFORMAÇÕES DO MODELO ===\n")
        f.write(f"Dimensões da malha: NX={NX}, NY={NY}, NZ={NZ}\n")
        f.write(f"dx: {dx}\n")
        f.write(f"dy: {dy}\n")
        f.write(f"dz: {dz}\n")
        f.write(f"Altura base (h): {h} m\n")
        f.write(f"Viscosidade de referência (mu_ref): {mu_ref} cP\n")
        f.write(f"Densidade de referência (rho_ref): {rho_ref} kg/m³\n")
        f.write(f"Pressão de referência (p_ref): {p_ref} kPa\n")
        f.write(f"Coef. viscosidade (c_mu): {c_mu} cP/kPa\n")
        f.write(f"Compressibilidade do fluido (c_f): {c_f} 1/kPa\n")
        f.write(f"Compressibilidade da rocha (c_r): {c_r} 1/kPa\n\n")

        f.write("=== RESULTADOS DOS POÇOS ===\n")
        f.write(f"{'k':>2} {'i':>3} {'j':>3} {'tipo':<10} {'controle':<10} {'pressao_kPa':>12} {'vazao_m3_dia':>14}\n")
        for k, i, j, tipo, controle, pres, q in well_results:
            f.write(f"{k:2d} {i:3d} {j:3d} {tipo:<10} {controle:<10} {pres:12.2f} {q:14.2f}\n")

        # Opcional: salvar pressão e propriedades
        if NX * NY <= 100:  # Limita exportação se malha for muito grande
            f.write("\n=== MAPA DE PRESSÃO POR CAMADA ===\n")
            for k in range(NZ):
                f.write(f"\n-- Camada {k} --\n")
                for row in P[k]:
                    f.write(" ".join(f"{val:7.2f}" for val in row) + "\n")

            f.write("\n=== POROSIDADE POR CAMADA ===\n")
            for k in range(NZ):
                f.write(f"\n-- Camada {k} --\n")
                for row in poro[k]:
                    f.write(" ".join(f"{val:6.4f}" for val in row) + "\n")

    print(f"Arquivo salvo em: {output_path}")
