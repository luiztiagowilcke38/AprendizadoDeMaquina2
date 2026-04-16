"""
Análise da Dívida Pública via Algoritmo LARS
Baseado no Cap. 48 do livro 'Aprendizado de Máquina' (Least Angle Regression).

Este script utiliza o algoritmo LARS para identificar automaticamente os 
principais vetores macroeconômicos do crescimento da dívida pública,
demonstrando o conceito de seleção esparsa de variáveis em dados tabulares.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars
from sklearn.preprocessing import StandardScaler
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

# 1. SIMULAÇÃO DE CENÁRIOS MACROECONÔMICOS
def generate_economic_dataset(n_samples=300):
    """
    Simula uma economia com indicadores reais e ruídos estatísticos.
    """
    np.random.seed(42)
    
    # --- Preditores Reais (Determinantes da Dívida) ---
    cresc_pib = np.random.normal(0.015, 0.012, n_samples)
    taxa_selic = np.random.normal(0.10, 0.03, n_samples)
    inflacao_ipca = np.random.normal(0.045, 0.02, n_samples)
    superavit_primario = np.random.normal(-0.01, 0.008, n_samples) # Negativo = Déficit
    cambio_usd = np.random.normal(5.0, 0.5, n_samples)
    
    # --- Preditores Irrelevantes (Ruído de dados) ---
    noise_vars = np.random.normal(0, 1, (n_samples, 15))
    
    # Target: Variação da Dívida Pública / PIB
    # A dívida cresce com juros altos (SELIC) e déficits (superávit negativo)
    # A dívida é amortecida pelo crescimento do PIB
    y = (4.0 * taxa_selic) + (2.5 * inflacao_ipca) - (6.0 * superavit_primario) - (2.0 * cresc_pib) + (0.5 * cambio_usd)
    y += np.random.normal(0, 0.02, n_samples) # Ruído de medição
    
    X = np.column_stack([cresc_pib, taxa_selic, inflacao_ipca, superavit_primario, cambio_usd, noise_vars])
    feature_names = ["Cresc_PIB", "Taxa_SELIC", "Inflacao_IPCA", "Result_Primario", "Taxa_Cambio"] + [f"Indice_{i+1}" for i in range(15)]
    
    return X, y, feature_names

# 2. ALGORITMO LARS (Least Angle Regression - Cap 48)
def run_debt_analysis():
    print("Gerando série histórica macroeconômica simulada...")
    X, y, features = generate_economic_dataset()
    
    # O LARS opera sobre dados centralizados e escalados (Cap 48.5)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # LARS encontra o caminho de otimização selecionando uma variável por vez
    lars = Lars(n_nonzero_coefs=500) # Usar um inteiro grande em vez de np.inf
    lars.fit(X_std, y)
    
    # Extrair o caminho dos coeficientes conforme evoluem na direção do 'ângulo mínimo'
    coef_path = lars.coef_path_
    
    # --- VISUALIZAÇÃO DO CAMINHO LARS ---
    os.makedirs("figuras", exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Eixo X: Norma L1 dos coeficientes (progressão do algoritmo)
    xx = np.sum(np.abs(coef_path.T), axis=1)
    xx /= xx[-1] # Normalizar para [0, 1]
    
    # Plotar os coeficientes de cada variável
    for i in range(X_std.shape[1]):
        # Destacar as 5 variáveis reais
        is_real = i < 5
        color = plt.cm.tab10(i) if is_real else 'lightgray'
        alpha = 1.0 if is_real else 0.4
        lw = 2.5 if is_real else 1.0
        
        plt.plot(xx, coef_path[i], label=features[i] if is_real else None,
                 color=color, alpha=alpha, lw=lw)
        
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.title("O Caminho LARS: Seleção de Variáveis da Dívida Pública\nIdentificação automática dos nós críticos da economia", fontsize=14)
    plt.xlabel("Norma L1 dos Coeficientes (Complexidade do Modelo $\\to$)", fontsize=12)
    plt.ylabel("Impacto Estimado (Beta)", fontsize=12)
    plt.legend(loc='upper left', ncol=2, frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig("figuras/public_debt_lars_analysis.png", dpi=300)
    plt.close()
    
    # 3. RELATÓRIO DE CONVERGÊNCIA
    print("\n" + "="*50)
    print("VETORES DE CRESCIMENTO DA DÍVIDA (Ordem de Seleção):")
    # Descobrir a ordem em que as variáveis entram no modelo
    entry_steps = []
    for i in range(X.shape[1]):
        first_nonzero_idx = np.where(coef_path[i] != 0)[0]
        if len(first_nonzero_idx) > 0:
            entry_steps.append((first_nonzero_idx[0], features[i]))
    
    # Ordenar por passo de entrada
    entry_steps.sort()
    for rank, (step, name) in enumerate(entry_steps[:6]):
        print(f"{rank+1}º Lugar: {name} (Passo LARS {step})")
        
    print("="*50)
    print("Figura gerada com sucesso: 'figuras/public_debt_lars_analysis.png'")

if __name__ == "__main__":
    run_debt_analysis()
