## Simulador de Escoamento Monofásico 3D com Propriedades Variáveis

Trabalho 02 da disciplina MAT2490  
Aluna Isabel Gonçalves - 2312237

Este simulador modela o escoamento monofásico em um meio poroso tridimensional (3D), considerando **viscosidade, densidade e porosidade dependentes da pressão**.

### Fundamentos Matemáticos

O modelo se baseia na **Lei de Darcy generalizada** e na conservação de massa. A equação governante é:

$$
\nabla \cdot \mathbf{q} = q_s
$$
$$
\mathbf{q} = -\frac{\mathbf{K}}{\mu(p)} \nabla p
$$

onde:
- $\mathbf{q}$: fluxo volumétrico (m³/dia)
- $\mu(p)$: viscosidade (função da pressão)
- $\mathbf{K}$: tensor de permeabilidade com anisotropia (kx, ky, kz)
- $p$: pressão
- $q_s$: termo-fonte (poços)

## Propriedades Dependentes da Pressão

As propriedades fluido-rocha variam com a pressão segundo as expressões:

- $\mu(p) &= \mu_{\text{ref}} \cdot e^{c_{\mu}(p - p_{\text{ref}})} $
- $\rho(p) &= \rho_{\text{ref}} \cdot \left(1 + c_f(p - p_{\text{ref}})\right)$
- $\phi(p) &= \phi_{\text{ref}} \cdot \left(1 + c_r(p - p_{\text{ref}})\right)$

## Discretização

A discretização é feita por diferenças finitas em malha estruturada 3D, com transmissibilidades calculadas usando médias harmônicas para cada direção (X, Y e Z). As transmissibilidades verticais conectam diretamente camadas adjacentes $k \pm 1$.

A equação resulta em sistemas lineares do tipo:

$$
\mathbf{T}^{(k)} \cdot \mathbf{P}^{(k)} = \mathbf{Q}^{(k)}
$$

resolvidos plano a plano, com acoplamento vertical sendo incorporado aos termos de borda superior e inferior.

## Poços

Poços podem ser especificados com controle por pressão ou vazão, agora com localização completa (i, j, **k**). O índice do plano perfurado é considerado nos cálculos de acoplamento e `Well Index`.

A vazão é computada como:

$$
q = \frac{WI \cdot (p_{\text{poço}} - p_{\text{bloco}})}{\mu(p)}
$$

com WI calculado conforme permeabilidades anisotrópicas e raio efetivo.

---

## Funcionalidades

- Malha 3D com dimensões arbitrárias
- Suporte a zonas inativas (via `grid.txt`)
- Propriedades físicas variáveis com a pressão
- Cálculo volumétrico tridimensional
- Visualização por camada (`P[k]`) ou múltiplos planos
- Exportação de mapas de pressão e permeabilidade
- Tabela de resultados por poço

---

## Arquivos de Entrada

* `input.txt`: parâmetros do problema:
```
[NX NY NZ]
dimensões da malha 3D
[DX] [DY]
tamanhos das células em X e Y (aceita 60*10 etc)
[DZ]
espessuras por camada (ex: 3*1.0)
[PROPS]
mu_ref rho_ref phi_ref
[REF]
p_ref c_mu c_r c_f
[WELLS]
número de poços
tipo raio i j k controle valor
```

* `grid.txt`: células ativas (3D, flatten por camada)
* `perm_x.txt`, `perm_y.txt`, `perm_z.txt`: permeabilidades em cada direção (uma linha por camada)
* `poro.txt`: porosidade por célula (linha por camada)
* `press.txt`: pressão inicial por célula

---

## Saídas

- Campo de pressão 3D (`P[k, i, j]`)
- Tabela por poço com:
  - localização `(i, j, k)`
  - tipo, controle, pressão e vazão simulada
- Imagens:
  - `mapa_pressao_k{n}.png`: pressão por camada
  - `mapa_permeabilidade.png`: mapas `kx`, `ky` com poços
  - `mapa_pressao_com_valores.png`
- Arquivo `resultados.txt` com todos os dados numéricos e parâmetros utilizados

---

## Scripts

- `simulation.py`: executa o simulador
- `simulation_functions.py`: funções de leitura, propriedades, solver
- `plot_functions.py`: geração de gráficos e mapas
