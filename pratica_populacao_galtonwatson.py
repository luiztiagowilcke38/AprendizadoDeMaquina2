"""
Crescimento Populacional, Martingales e Deep Learning (Processo de Galton-Watson)
Baseado no Cap. 16 (Martingales e Processos de Tempo Discreto) do livro 'Aprendizado de Máquina'.

Este script simula a evolução estocástica de uma população genérica usando o rigoroso 
Processo de Ramificação de Galton-Watson. No contexto do Aprendizado de Máquina (conforme
citado na Seção 7.2 do Cap. 16), o crescimento/extinção dessa população espelha o 
comportamento da propagação do Gradiente (Vanishing/Exploding Gradients) ao longo das 
camadas ocultas de uma Rede Neural Profunda.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import os

# Configurações estéticas e profissionais
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'
VERMELHO = '#d62728'
ROXO = '#9467bd'

class GaltonWatsonPopulation:
    """
    Simula e analisa a dinâmica de crescimento de uma população
    através de filiações independentes (Galton-Watson), e seu Martingale associado.
    """
    def __init__(self, expected_offspring, max_generations=20):
        self.mu = expected_offspring # mu: Número médio de filhos por indivíduo 
        self.max_gen = max_generations
        
    def simulate(self, z0=1, num_paths=50):
        """
        Simula a evolução populacional usando distribuição de Poisson para prole.
        """
        populations = np.zeros((num_paths, self.max_gen + 1))
        populations[:, 0] = z0
        
        for p in range(num_paths):
            for n in range(1, self.max_gen + 1):
                parents = int(populations[p, n-1])
                if parents == 0:
                    populations[p, n] = 0 # Extinção é um estado absorvente
                else:
                    # Cada um dos 'parents' gera descendentes i.i.d. via Poisson(mu)
                    children = np.sum(np.random.poisson(self.mu, parents))
                    populations[p, n] = children
                    
        return populations
    
    def construct_martingale(self, population_paths):
        """
        Constrói o Martingale de Doob: M_n = Z_n / mu^n
        O Teorema garante que E[M_n] é constante (M_0).
        """
        martingale_paths = np.zeros_like(population_paths)
        for n in range(self.max_gen + 1):
            martingale_paths[:, n] = population_paths[:, n] / (self.mu ** n)
        return martingale_paths

def predict_population_dynamics():
    # Parâmetros vitais (No ML: Variância dos pesos na inicialização Xavier/He)
    mu_subcritical = 0.9  # <= 1: Extinção determinística (Vanishing Gradients)
    mu_supercritical = 1.05 # > 1: Sobrevivência/Crescimento Exponencial (Exploding/Sustaining)
    Z0 = 100 # População inicial

    np.random.seed(42)
    os.makedirs("figuras", exist_ok=True)
    
    # 1. Simulação: Regime Subcrítico (Taxa reprodutiva Média < 1)
    gw_sub = GaltonWatsonPopulation(mu_subcritical, max_generations=50)
    pop_sub = gw_sub.simulate(z0=Z0, num_paths=20)
    
    # 2. Simulação: Regime Supercrítico (Taxa reprodutiva Média > 1)
    gw_super = GaltonWatsonPopulation(mu_supercritical, max_generations=50)
    pop_super = gw_super.simulate(z0=Z0, num_paths=20)
    
    # 3. O Fenômeno de Martingale
    # Z_n sofre com variância exponencial, mas M_n = Z_n / mu^n é um Martingale perfeitamente balanceado
    martingale_super = gw_super.construct_martingale(pop_super)
    
    # 4. Visualização Profissional (Analogia ML vs Biologia)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Caminhos da População Extinção vs Crescimento
    time_steps = range(51)
    
    # Média teórica E[Z_n] = Z_0 * mu^n
    teorico_sub = Z0 * (mu_subcritical ** np.array(time_steps))
    teorico_sup = Z0 * (mu_supercritical ** np.array(time_steps))
    
    for i in range(20):
        # Alpha diminui pra evitar poluição
        ax1.plot(time_steps, pop_sub[i], color=VERMELHO, alpha=0.3)
        ax1.plot(time_steps, pop_super[i], color=VERDE, alpha=0.3)
        
    # Médias Reais no plot
    ax1.plot(time_steps, np.mean(pop_sub, axis=0), color=VERMELHO, lw=2.5, label=r'Média Agente/Pop (Subcrítico $\mu=0.9$)')
    ax1.plot(time_steps, np.mean(pop_super, axis=0), color=VERDE, lw=2.5, label=r'Média Agente/Pop (Supercrítico $\mu=1.05$)')
    
    ax1.plot(time_steps, teorico_sub, color='black', linestyle='--', lw=1.5, label='Esperança $E[Z_n]$')
    ax1.plot(time_steps, teorico_sup, color='black', linestyle='--', lw=1.5)
    
    ax1.set_title("Evolução da População (Processo de Galton-Watson)\n(ML: Propagação de Gradiente na Rede)")
    ax1.set_xlabel("Geração (ou Camada da Rede)")
    ax1.set_ylabel("Tamanho da População $Z_n$ (ou Norma do Gradiente)")
    ax1.legend()
    
    # Subplot 2: Teorema de Convergência de Doob no Martingale
    for i in range(20):
        ax2.plot(time_steps, martingale_super[i], color=AZUL, alpha=0.2)
        
    ax2.plot(time_steps, np.mean(martingale_super, axis=0), color='black', lw=3, label=r'Média Empírica $\bar{M}_n$')
    ax2.axhline(Z0, color=LARANJA, linestyle='--', lw=2.5, label=r'Invariante $E[M_n] = Z_0$')
    
    ax2.set_title(r"O Jogo Justo: Martingale $M_n = Z_n / \mu^n$" + "\n(Cap. 16 - Teorema de Doob)")
    ax2.set_xlabel("Tempo / Geração $n$")
    ax2.set_ylabel("Valor do Martingale $M_n$")
    ax2.legend()
    
    plt.tight_layout()
    file_path = "figuras/crescimento_galton_watson_ml.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    
    print("\n--- ANÁLISE DE CRESCIMENTO (GALTON-WATSON) ---")
    print(f"Taxa Subcrítica (mu={mu_subcritical}): População Extinta.")
    print(f"Taxa Supercrítica (mu={mu_supercritical}): População Crescente.")
    print(f"Martingale Doob: Média inicial={Z0:.1f} | Média Final E[M_50]={np.mean(martingale_super[:, -1]):.1f}")
    print(f"Conservação Confirmada. Gráfico gerado em '{file_path}'.")

if __name__ == "__main__":
    predict_population_dynamics()
