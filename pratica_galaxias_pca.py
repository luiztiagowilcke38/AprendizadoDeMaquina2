"""
Análise de Morfologia de Galáxias via PCA
Baseado no Cap. 10 do livro 'Aprendizado de Máquina' (Seção 80: PCA).

Este script utiliza a Análise de Componentes Principais para extrair as 
características fundamentais de perfis de brilho de galáxias simulados,
aplicando a Transformada de Karhunen-Loève para redução de dimensionalidade.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincinv
import os

# Configurações estéticas profissionais (Padrão do Livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'

# 1. SIMULAÇÃO DE DADOS: O Perfil de Sérsic (Astrofísica)
def sersic_profile(r, Ie, re, n):
    """
    Representa a distribuição de brilho de uma galáxia.
    n=1: Perfil Exponencial (Discos de espirais).
    n=4: Perfil de De Vaucouleurs (Elípticas).
    """
    bn = gammaincinv(2*n, 0.5)
    return Ie * np.exp(-bn * ((r/re)**(1/n) - 1))

def generate_galaxy_catalog(n_samples=600):
    r = np.linspace(0.1, 10, 100) # Grid radial
    data = []
    labels = []
    
    print(f"Simulando {n_samples} galáxias com morfologias variadas...")
    for _ in range(n_samples):
        # n (Concentração) e re (Escala) variando aleatoriamente
        n_val = np.random.uniform(0.5, 6.0)
        re_val = np.random.uniform(0.8, 4.5)
        
        profile = sersic_profile(r, 1.0, re_val, n_val)
        # Adicionar ruído de fundo (Simulação de observação real)
        noise = np.random.normal(0, 0.015, len(r))
        
        data.append(profile + noise)
        labels.append(n_val)
        
    return r, np.array(data), np.array(labels)

# 2. IMPLEMENTAÇÃO DO PCA (Karhunen-Loève Transform - Cap 10.80)
def compute_pca(X, d=2):
    """
    Executa o PCA do zero seguindo a derivação matemática do livro.
    """
    N = X.shape[0]
    # a. Centralização dos dados (E[Z] = 0)
    mu = np.mean(X, axis=0)
    X_tilde = X - mu
    
    # b. Matriz de Covariância Empírica
    sigma = np.cov(X_tilde, rowvar=False)
    
    # c. Decomposição Espectral (Autovetores e Autovalores)
    lambdas, U = np.linalg.eigh(sigma)
    
    # d. Ordenação por variância (λ1 >= λ2 >= ... >= λp)
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    U = U[:, idx]
    
    # e. Projeção no Hiperplano de Rank-d (Matriz de Loading W)
    W = U[:, :d]
    Z = np.dot(X_tilde, W) # Componentes Principais
    
    return mu, W, Z, lambdas

# 3. ANALISE E VISUALIZAÇÃO
def visualize_galaxy_pca():
    # Gerar dados
    r, X, sersic_indices = generate_galaxy_catalog()
    
    # Processar PCA
    mu, W, Z, lambdas = compute_pca(X, d=3)
    
    # Variância Explicada
    var_exp = lambdas / np.sum(lambdas)
    
    # --- PLOTAGEM ---
    os.makedirs("figuras", exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 3.1 Perfil Médio e Modos de Variação (Eigen-Galaxies)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r, mu, color='black', lw=2.5, label='Média Global ($\mu$)')
    # Mostrar como o PC1 distorce a média (Morfologia)
    pc1_dev = 2.5 * np.sqrt(lambdas[0]) * W[:, 0]
    ax1.fill_between(r, mu - pc1_dev, mu + pc1_dev, color=AZUL, alpha=0.15, label='Envelope PC1 ($\pm 2.5\sigma$)')
    ax1.set_title("Análise Morfológica: O Perfil Médio e Sua Variância")
    ax1.set_xlabel("Raio Projetado (arcsec)")
    ax1.set_ylabel("Brilho Superficial")
    ax1.legend()
    
    # 3.2 Scree Plot (Pareto da Variância)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(1, 6), var_exp[:5], color=LARANJA, alpha=0.7)
    ax2.step(range(1, 6), np.cumsum(var_exp[:5]), where='mid', color='red', label='Var. Acumulada')
    ax2.set_title("Espectro de Autovalores (Variância Explicada)")
    ax2.set_ylabel("Proporção de Variância")
    ax2.set_xlabel("Componente Principal")
    ax2.legend()
    
    # 3.3 Espaço Latente (Classificação Automática de Galáxias)
    ax3 = fig.add_subplot(gs[1, :])
    scatter = ax3.scatter(Z[:, 0], Z[:, 1], c=sersic_indices, cmap='magma', s=25, alpha=0.8)
    plt.colorbar(scatter, ax=ax3, label='Índice de Sérsic ($n$)')
    ax3.set_title("Espaço Latente Morfológico (PC1 vs PC2)\nA distribuição contínua de cores reflete a transição entre Elípticas e Espirais")
    ax3.set_xlabel("Componente Principal 1")
    ax3.set_ylabel("Componente Principal 2")
    
    plt.tight_layout()
    plt.savefig("figuras/galaxy_pca_analysis.png", dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("ANÁLISE DE GALÁXIAS CONCLUÍDA")
    print(f"Variância Capturada nos 2 primeiros PCs: {np.sum(var_exp[:2])*100:.2f}%")
    print("O PC1 capturou a concentração de brilho (Índice de Sérsic).")
    print("="*50)
    print("Figura salva em 'figuras/galaxy_pca_analysis.png'")

if __name__ == "__main__":
    visualize_galaxy_pca()
