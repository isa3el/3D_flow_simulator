## Simulador de Escoamento Monofásico 3D

Trabalho 02 da disciplina MAT2490  
Aluna Isabel Gonçalves - 2312237

Este simulador modela o escoamento monofásico em um meio poroso tridimensional (3D), considerando **viscosidade, densidade e porosidade dependentes da pressão**, além de **acoplamento vertical entre camadas**.

### Fundamentos Matemáticos

A equação governante do modelo combina a **Lei de Darcy generalizada** com a **conservação de massa**,  resultando na seguinte EDP:

$$
\frac{\partial (\phi \rho)}{\partial t} - \nabla \cdot \left( \frac{\rho \mathbf{k}}{\mu} (\nabla p + \rho \mathbf{g}) \right) - (\rho q) = 0
$$

com:
- $\phi = \phi(p)$: porosidade variável com a pressão
- $\rho = \rho(p)$: densidade variável com a pressão
- $\mu = \mu(p)$: viscosidade variável com a pressão
- $\mathbf{k}$: tensor de permeabilidade
- $\mathbf{g}$: vetor gravidade
- $q$: termo-fonte (poços)

## Propriedades Dependentes da Pressão

As propriedades fluido-rocha variam com a pressão segundo as expressões:

$$\mu(p) = \mu_{\text{ref}} \cdot e^{c_{\mu}(p - p_{\text{ref}})}$$
$$\rho(p) = \rho_{\text{ref}} \cdot \left(1 + c_f(p - p_{\text{ref}})\right)$$
$$\phi(p) = \phi_{\text{ref}} \cdot \left(1 + c_r(p - p_{\text{ref}})\right)$$


### Discretização

Aplicando o formalismo do MVF com tratamento implícito dos termos temporais, o balanço de massa em uma célula $i$ é dado por:

$$
V_i \cdot \frac{(\phi \rho)^{n+1}_i - (\phi \rho)^n_i}{\Delta t^n} + \sum_{k \in \chi_\ell} \lambda_{ik}^{n+1} T_{ik} \left(p_i^{n+1} - p_k^{n+1} + \rho_{ik}^{n+1}(z_i - z_k)\right) - \lambda_{w_i}^{n+1}WI_i(p_i^{n+1} - p_{wf_k}^{n+1}) = 0
$$


Onde os termos representam:
- **Termo temporal:** 
$$V_i \cdot \frac{(\phi \rho)^{n+1}_i - (\phi \rho)^n_i}{\Delta t}$$
- **Fluxo com gravidade:** 
$$\sum_{k \in \chi_i} \lambda_{ik}^{n+1} T_{ik} (p_i^{n+1} - p_k^{n+1} + \rho g (z_i - z_k))$$
- **Termo dos poços:** 
$$-\lambda_w \cdot WI \cdot (p_i - p_{wf})$$

A matriz $\mathbf{T}$ é montada globalmente, considerando acoplamento completo entre camadas.

Essa equação resulta em um sistema linear para as pressões desconhecidas no tempo $n+1$, resolvido a cada passo de tempo por um método direto. 

### Atenção: 
No início de `simulation.py` é possível definir o número de passos de tempo (`n_steps`) a serem simulados. No entanto, valores de `n_steps` > 5 provocam instabilidades numéricas, levando a divergência na solução. Observei que a diferença de pressão entre blocos vizinhos passa a conter valores NaN, o que desencadeia um efeito dominó na simulação: a matriz de transmissibilidades se torna inconsistente, resultando em pressões irreais (valores astronômicos) ao longo do tempo. Essa instabilidade se reflete diretamente nos resultados salvos em results_3D.txt, onde a pressão em alguns poços explode para valores fisicamente inviáveis. O problema se agrava com o aumento de `n_steps`

### Poços


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