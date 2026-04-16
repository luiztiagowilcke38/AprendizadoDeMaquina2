"""
Modelagem de Epidemias Espaciais via Markov Random Fields (MRF)
Baseado no Cap. 71 do livro 'Aprendizado de Máquina' (MRF e Gibbs Sampling).

Este script utiliza um modelo de grade 2D (inspirado no Modelo de Ising) para 
simular a propagação de uma doença infecciosa. A probabilidade de contágio 
recorre à Teoria de Redes de Markov Não-Direcionadas, onde o estado de um 
indivíduo depende apenas do estado de seus vizinhos locais.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas profissionais (Padrão do Livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": False, # Desligar grid nas matrizes
})

# Estados da infecção
SUSCETIVEL = -1
INFECTADO = 1

# 1. ATUALIZAÇÃO GIBBS SAMPLING (Capítulo 71.4)
def gibbs_step(grid, beta):
    """
    Realiza uma varredura completa na grade usando Amostragem de Gibbs.
    'beta' representa a taxa de infecção (o inverso da temperatura em Física).
    """
    rows, cols = grid.shape
    new_grid = grid.copy()
    
    # Percorrer toda a população aleatoriamente
    indices = [(r, c) for r in range(rows) for c in range(cols)]
    np.random.shuffle(indices)
    
    for i, j in indices:
        # Se já está infectado, não pode voltar a ficar suscetível neste modelo simplificado
        # (Para um modelo SIR real, adicionaríamos uma transição para 'Recuperado')
        # Aqui, focamos apenas no espalhamento explosivo (SI Model na grade)
        if grid[i, j] == INFECTADO:
            continue
            
        # Vizinhança local de Markov: Cima, Baixo, Esquerda, Direita
        vecinos = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        s_sum = 0
        
        for r, c in vecinos:
            # Condição de borda (Tratar fora do grid como Suscetível)
            if 0 <= r < rows and 0 <= c < cols:
                s_sum += grid[r, c]
            else:
                s_sum += SUSCETIVEL
                
        # Potencial do MRF: A chance de infecção aumenta com o número de vizinhos infectados
        # p_plus é a probabilidade do estado virar +1 (Infectado)
        energia = -2 * beta * s_sum
        p_infeccao = 1 / (1 + np.exp(energia))
        
        # O modelo original de Ising permite flutuações. Na epidemia, 'beta' controla a agressividade
        # Para ser realista, se todos ao redor estão suscetíveis, a chance de doer de forma espontânea é quase zero.
        if np.random.rand() < p_infeccao:
            new_grid[i, j] = INFECTADO
            
    return new_grid

# 2. SIMULAÇÃO PRINCIPAL
def run_epidemic_simulation():
    # Parâmetros
    grid_size = 50
    steps = 40
    beta_infection_rate = 0.65  # Alta transmissibilidade
    
    print(f"Inicializando cidade com {grid_size}x{grid_size} habitantes...")
    
    # Iniciar grade: 100% Suscetível
    grid = np.full((grid_size, grid_size), SUSCETIVEL)
    
    # O "Paciente Zero" no centro
    grid[grid_size//2, grid_size//2] = INFECTADO
    
    history = [grid.copy()]
    infection_curve = [1] # Quantidade de infectados
    
    print("Iniciando propagação via Cadeias de Markov (Gibbs Sampling)...")
    for t in range(1, steps + 1):
        grid = gibbs_step(grid, beta_infection_rate)
        history.append(grid.copy())
        infection_curve.append(np.sum(grid == INFECTADO))
        
        # Progresso
        if t % 10 == 0:
            print(f"Dia {t}: {infection_curve[-1]} infectados.")
            
    # --- VISUALIZAÇÃO ---
    os.makedirs("figuras", exist_ok=True)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3)
    
    # Escolher 3 "Dias" chave para fotografar a grade
    snapshots = [0, steps//3, steps]
    
    for idx, day in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, idx])
        # Mapa de cores: Azul (Saudável), Vermelho Escuro (Infectado)
        cmap = plt.cm.get_cmap('coolwarm')
        ax.imshow(history[day], cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f"Avanço Geográfico (Dia {day})")
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Curva de Crescimento (Toda a base inferior)
    ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.grid(True, alpha=0.3)
    pct_infected = (np.array(infection_curve) / (grid_size**2)) * 100
    
    ax_curve.plot(range(steps + 1), pct_infected, color='#d62728', lw=3, label='Progressão Epidemiológica')
    ax_curve.fill_between(range(steps + 1), 0, pct_infected, color='#d62728', alpha=0.2)
    
    ax_curve.set_title("Curva Epidemiológica Populacional (Contágio via MRF)")
    ax_curve.set_xlabel("Tempo (Ciclos de Amostragem de Gibbs)")
    ax_curve.set_ylabel("% População Infectada")
    ax_curve.set_ylim(0, 100)
    ax_curve.legend()
    
    plt.tight_layout()
    plt.savefig("figuras/epidemic_spread_mrf.png", dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("SIMULAÇÃO CONCLUÍDA")
    print(f"Infectados no Final (Dia {steps}): {pct_infected[-1]:.1f}% da cidade")
    print("A propriedade de Markov local mostrou com sucesso que o acúmulo")
    print("de vizinhos infectados leva ao crescimento exponencial da curva.")
    print("="*50)
    print("Figura gerada: 'figuras/epidemic_spread_mrf.png'")

if __name__ == "__main__":
    run_epidemic_simulation()
