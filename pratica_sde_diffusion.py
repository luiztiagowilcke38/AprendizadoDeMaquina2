"""
Framework Avançado de Cálculo Estocástico e Modelos de Difusão
Baseado no Cap. 17 (Cálculo Estocástico) e Cap. 10 (Não Supervisionado) do livro 'Aprendizado de Máquina'.

Este script implementa solvers para SDEs (Equações Diferenciais Estocásticas) do zero,
demonstrando a matemática por trás dos modelos de difusão modernos (DDPM, Score-based models).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Criar diretorio para figuras se não existir
os.makedirs("figuras", exist_ok=True)

# Configurações estéticas profissionais (Seguindo o padrão do livro)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'
ROXO = '#9467bd'

# =============================================================================
# 1. SOLVERS PARA SDE (EQUAÇÕES DIFERENCIAIS ESTOCÁSTICAS)
# =============================================================================

class SDESolver:
    """Classe base para solvers de SDE: dX = b(X, t)dt + sigma(X, t)dB"""
    def __init__(self, drift, diffusion, diffusion_prime=None):
        self.b = drift           # Coeficiente de Drift
        self.sigma = diffusion   # Coeficiente de Difusão
        self.sigma_p = diffusion_prime # Derivada da difusão (necessário para Milstein)

    def solve(self, x0, t_span, dt, n_paths=1):
        raise NotImplementedError

class EulerMaruyama(SDESolver):
    """
    Solver de Euler-Maruyama: Aproximação de 1ª ordem (convergência forte 0.5).
    X_{t+1} = X_t + b(X_t, t)*dt + sigma(X_t, t)*sqrt(dt)*Z
    """
    def solve(self, x0, t_span, dt, n_paths=1):
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)
        paths = np.zeros((n_steps, n_paths))
        paths[0] = x0
        
        for i in range(1, n_steps):
            z = np.random.randn(n_paths)
            dw = np.sqrt(dt) * z
            paths[i] = paths[i-1] + self.b(paths[i-1], t[i-1]) * dt + \
                       self.sigma(paths[i-1], t[i-1]) * dw
        return t, paths

class Milstein(SDESolver):
    """
    Solver de Milstein: Inclui termo de 2ª ordem para difusão dependente do estado.
    Convergência forte de 1.0.
    """
    def solve(self, x0, t_span, dt, n_paths=1):
        if self.sigma_p is None:
            print("Aviso: Milstein requer sigma_prime. Usando Euler-Maruyama.")
            return EulerMaruyama(self.b, self.sigma).solve(x0, t_span, dt, n_paths)
            
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)
        paths = np.zeros((n_steps, n_paths))
        paths[0] = x0
        
        for i in range(1, n_steps):
            z = np.random.randn(n_paths)
            dw = np.sqrt(dt) * z
            # Termo de Milstein: 0.5 * sigma * sigma' * (dw^2 - dt)
            correction = 0.5 * self.sigma(paths[i-1], t[i-1]) * \
                         self.sigma_p(paths[i-1], t[i-1]) * (dw**2 - dt)
            
            paths[i] = paths[i-1] + self.b(paths[i-1], t[i-1]) * dt + \
                       self.sigma(paths[i-1], t[i-1]) * dw + correction
        return t, paths

# =============================================================================
# 2. DEFINIÇÃO DE PROCESSOS CLÁSSICOS
# =============================================================================

def get_ornstein_uhlenbeck(theta=1.0, mu=0.0, sigma=0.5):
    """Processo de Reversão à Média (Cap. 17.8)"""
    drift = lambda x, t: -theta * (x - mu)
    diffusion = lambda x, t: sigma
    return drift, diffusion

def get_geometric_brownian(mu=0.1, sigma=0.2):
    """Movimento Browniano Geométrico (Crescimento Exponencial Estocástico)"""
    drift = lambda x, t: mu * x
    diffusion = lambda x, t: sigma * x
    diffusion_p = lambda x, t: sigma
    return drift, diffusion, diffusion_p

# =============================================================================
# 3. SCORE-BASED MODELS & LANGEVIN DYNAMICS (Cap. 10 / 17)
# =============================================================================

def langevin_dynamics(score_fn, x_start, n_steps=100, eps=0.01):
    """
    Amostragem via Dinâmica de Langevin: Amostra de P(x) usando apenas o Score.
    x_{t+1} = x_t + (eps/2) * grad(log P(x_t)) + sqrt(eps) * Z
    """
    x = x_start.copy()
    history = [x.copy()]
    for _ in range(n_steps):
        z = np.random.randn(*x.shape)
        x = x + (eps / 2.0) * score_fn(x) + np.sqrt(eps) * z
        history.append(x.copy())
    return np.array(history)

def mixture_score(x):
    """Score function para uma mistura de duas Gaussianas"""
    # P(x) = 0.5*N(-2, 0.5) + 0.5*N(2, 0.5)
    # Score = d/dx log P(x) = (1/P(x)) * d/dx P(x)
    m1, s1 = -2.5, 0.6
    m2, s2 = 2.5, 0.6
    
    p1 = norm.pdf(x, m1, s1)
    p2 = norm.pdf(x, m2, s2)
    p_total = 0.5 * p1 + 0.5 * p2
    
    grad_p1 = p1 * (-(x - m1) / s1**2)
    grad_p2 = p2 * (-(x - m2) / s2**2)
    grad_total = 0.5 * grad_p1 + 0.5 * grad_p2
    
    return grad_total / (p_total + 1e-9)

# =============================================================================
# 4. EXECUÇÃO E VISUALIZAÇÃO
# =============================================================================

def plot_sde_results():
    print("Simulando processos estocásticos...")
    
    # 4.1 Ornstein-Uhlenbeck (Forward Diffusion)
    drift_ou, diff_ou = get_ornstein_uhlenbeck(theta=1.5, mu=0.0, sigma=0.8)
    solver = EulerMaruyama(drift_ou, diff_ou)
    t, paths = solver.solve(x0=2.0, t_span=(0, 5), dt=0.01, n_paths=100)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, paths[:, :5], alpha=0.8)
    plt.plot(t, np.mean(paths, axis=1), color='black', lw=2, label='Média Empírica')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label=r'Média Teórica ($\mu$)')
    plt.title("Processo de Ornstein-Uhlenbeck (Reversão à Média)\nUtilizado em Modelos de Difusão (Forward)")
    plt.xlabel("Tempo ($t$)")
    plt.ylabel("$X_t$")
    plt.legend()
    plt.savefig("figuras/ou_process.png", dpi=300)
    plt.close()

    # 4.2 Evolução da Densidade (Heat Equation)
    plt.figure(figsize=(10, 5))
    steps_to_plot = [0, 50, 100, 300, 499]
    for step in steps_to_plot:
        plt.hist(paths[step], bins=30, density=True, alpha=0.3, label=f't = {t[step]:.1f}')
    plt.title("Evolução da Densidade de Probabilidade (Equação de Fokker-Planck)")
    plt.xlabel("$X$")
    plt.ylabel("Densidade")
    plt.legend()
    plt.savefig("figuras/density_evolution.png", dpi=300)
    plt.close()

    # 4.3 Amostragem por Score (Langevin Dynamics)
    print("Executando Langevin Dynamics...")
    x_init = np.random.uniform(-5, 5, 200)
    samples_path = langevin_dynamics(mixture_score, x_init, n_steps=500, eps=0.05)
    final_samples = samples_path[-1]
    
    plt.figure(figsize=(10, 5))
    x_range = np.linspace(-6, 6, 400)
    p_true = 0.5 * norm.pdf(x_range, -2.5, 0.6) + 0.5 * norm.pdf(x_range, 2.5, 0.6)
    
    plt.plot(x_range, p_true, color='black', lw=2, label='Distribuição Alvo (Mistura)')
    plt.hist(final_samples, bins=40, density=True, color=AZUL, alpha=0.5, label='Amostras Langevin')
    plt.title("Amostragem via Langevin Dynamics (Reverse Diffusion)\nRecuperando P(x) a partir do Score Function $\\nabla \\log P(x)$")
    plt.legend()
    plt.savefig("figuras/langevin_sampling.png", dpi=300)
    plt.close()
    
    # 4.4 Comparação Euler vs Milstein (Convergência)
    print("Analisando convergência de Solvers...")
    b_gbm, s_gbm, sp_gbm = get_geometric_brownian(mu=0.05, sigma=0.3)
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    errors_euler = []
    errors_milstein = []
    
    # Solução analítica do GBM: X_t = X0 * exp((mu - 0.5*sigma^2)*t + sigma*Bt)
    # Para simplificar o teste de convergência forte, usamos o mesmo noise p/ Euler e Milstein
    # Aqui vamos apenas ilustrar a diferença de estabilidade em um path
    
    plt.figure(figsize=(10, 5))
    t_m, p_m = Milstein(b_gbm, s_gbm, sp_gbm).solve(1.0, (0, 2), 0.01, 1)
    t_e, p_e = EulerMaruyama(b_gbm, s_gbm).solve(1.0, (0, 2), 0.01, 1)
    # Nota: para comparação real, precisariam compartilhar o Wiener Increment exato.
    plt.plot(t_m, p_m, label='Milstein (Ordem 1.0)', color=AZUL)
    plt.plot(t_e, p_e, label='Euler (Ordem 0.5)', color=LARANJA, alpha=0.7)
    plt.title("Comparação de Solvers em Movimento Browniano Geométrico")
    plt.legend()
    plt.savefig("figuras/solver_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_sde_results()
    print("\nScript finalizado com sucesso!")
    print("As figuras foram salvas no diretorio 'figuras/'.")
    print("Este framework demonstra os conceitos fundamentais do Cap. 17 do seu livro.")
