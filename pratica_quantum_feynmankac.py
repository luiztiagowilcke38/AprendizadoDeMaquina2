"""
Física Quântica Estocástica: Fórmula de Feynman-Kac
Baseado no Cap. 17 do livro 'Aprendizado de Máquina' (Cálculo Estocástico).

Este script utiliza a conexão entre difusão estocástica e a equação de Schrödinger
para estimar o estado fundamental de um potencial quântico via caminhos brownianos,
demonstrando a dualidade entre física estatística e quântica.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas profissionais (Seguindo o padrão do livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'

# 1. DEFINIÇÃO DO POTENCIAL QUÂNTICO (Cap. 17.5)
def potential(x):
    """
    Potencial do Oscilador Harmônico Quântico: V(x) = 0.5 * m * w^2 * x^2.
    Em unidades atômicas (hbar = m = w = 1), V(x) = 0.5 * x^2.
    """
    return 0.5 * (x**2)

# 2. SOLVER DE FEYNMAN-KAC (INTEGRAL DE CAMINHO)
def feynman_kac_solver(x0_range, n_paths=2000, T=3.0, dt=0.01):
    """
    Resolve a equação de Schrödinger em tempo imaginário via caminhos aleatórios.
    A função de onda psi(x, T) é a esperança matemática dos caminhos brownianos
    ponderados pela exponencial da integral do potencial (Ação de Euclides).
    
    psi(x, T) = E_x [ exp(-int_0^T V(B_s) ds) ]
    """
    n_steps = int(T / dt)
    psi_estimates = []
    
    print(f"Propagando {n_paths} caminhos brownianos para cada ponto do grid...")
    
    for x0 in x0_range:
        # Simulação simultânea de n_paths iniciados em x0
        # Estado inicial X0 = x0 (Fórmula de Feynman-Kac requer caminhos partindo de x)
        paths = np.zeros((n_steps, n_paths))
        paths[0] = x0
        
        # Gerar incrementos brownianos dB ~ N(0, dt) (Cap. 17.1)
        increments = np.random.normal(0, np.sqrt(dt), (n_steps-1, n_paths))
        paths[1:] = x0 + np.cumsum(increments, axis=0)
        
        # Calcular a Integral de Itô/Riemann do Potencial ao longo do caminho
        # Esta integral representa a 'penalização' de energia do caminho
        v_values = potential(paths)
        v_integral = np.sum(v_values, axis=0) * dt
        
        # O peso estocástico de cada caminho é exp(-V_integral)
        weights = np.exp(-v_integral)
        
        # A estimativa local de psi(x) é a média desses pesos
        psi_estimates.append(np.mean(weights))
        
    return np.array(psi_estimates)

# 3. EXPERIMENTO E COMPARAÇÃO ANALÍTICA
def run_quantum_stochastic_simulation():
    # Grid espacial
    x_range = np.linspace(-3, 3, 60)
    
    # Parâmetros da simulação
    # T (tempo imaginário) atua como um 'filtro' que faz o sistema decair 
    # para o estado de menor energia (Ground State)
    T = 5.0 
    n_paths = 5000 # Quantidade de caminhos para reduzir o erro de Monte Carlo
    
    # Executar o Solver Estocástico
    psi_stochastic = feynman_kac_solver(x_range, n_paths=n_paths, T=T)
    
    # Normalizar para comparação (O Ground State real é uma Gaussiana normalizada)
    psi_stochastic /= np.max(psi_stochastic)
    
    # Solução Analítica do Oscilador Harmônico (Ground State): psi(x) = exp(-x^2 / 2)
    psi_analytical = np.exp(-x_range**2 / 2)
    
    # --- VISUALIZAÇÃO ---
    os.makedirs("figuras", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Plotar o Potencial de Confinamento
    plt.plot(x_range, potential(x_range), '--', color='gray', alpha=0.3, label='Potencial Harmônico V(x)')
    
    # Plotar Comparação
    plt.plot(x_range, psi_analytical, color='black', lw=2, label='Solução Analítica (Ground State)')
    plt.scatter(x_range, psi_stochastic, color=AZUL, alpha=0.6, s=15, 
                label='Estimativa via Feynman-Kac (Stochastic paths)')
    
    # Estética
    plt.title("Ground State Quântico via Caminhos Estocásticos (Feynman-Kac)\nDemonstrando a convergência da simulação de Monte Carlo")
    plt.xlabel("Posição (x)")
    plt.ylabel("Amplitude da Função de Onda $\psi(x)$")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Salvar
    plt.savefig("figuras/quantum_feynmankac_results.png", dpi=300)
    plt.close()
    
    # Estatísticas
    mse = np.mean((psi_stochastic - psi_analytical)**2)
    print("\n" + "="*50)
    print(f"CONVERGÊNCIA QUÂNTICA:")
    print(f"Erro Médio Quadrático (MSE): {mse:.8f}")
    print(f"Tempo Imaginário de Propagação (T): {T}")
    print(f"Número de Caminhos por Ponto: {n_paths}")
    print("="*50)
    print("Figura gerada com sucesso: 'figuras/quantum_feynmankac_results.png'")

if __name__ == "__main__":
    run_quantum_stochastic_simulation()
