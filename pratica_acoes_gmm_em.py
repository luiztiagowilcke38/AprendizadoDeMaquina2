"""
Aprendizado Não Supervisionado em Finanças: Detecção de Regimes de Mercado (GMM e Algoritmo EM)
Baseado no Cap. 27 (Algoritmo EM e Modelos de Mistura) e Cap. 10 do livro 'Aprendizado de Máquina'.

Este script implementa o Algoritmo Expectation-Maximization (EM) do zero para ajustar um
Modelo de Mistura de Gaussianas (GMM). A aplicação clássica na bolsa de valores é a 
identificação de regimes de mercado ocultos (Latent States), como "Bull Market" (Alta, baixa 
volatilidade) e "Bear Market" (Baixa, alta volatilidade), observando apenas a série de retornos.
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
VERMELHO = '#d62728'
VERDE = '#2ca02c'

class GaussianMixture1D_EM:
    """
    Implementação rigorosa do Algoritmo Expectation-Maximization para GMM (1D).
    Conforme derivado no Capítulo 27 do livro (Misturas de Gaussianas).
    """
    def __init__(self, k=2, max_iter=100, tol=1e-6, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        np.random.seed(random_state)
        
        # Parâmetros a serem aprendidos
        self.mu = None
        self.sigma2 = None
        self.pi = None # Pesos da mistura
        self.responsibilities = None
        self.log_likelihoods = []
        
    def _pdf(self, x, mu, sigma2):
        """Função densidade de probabilidade da Normal."""
        return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * ((x - mu)**2) / sigma2)
        
    def fit(self, X):
        N = len(X)
        
        # Inicialização: K-Means relaxado (atribuição aleatória e estimação inicial)
        self.mu = np.random.choice(X, self.k)
        self.sigma2 = np.ones(self.k) * np.var(X)
        self.pi = np.ones(self.k) / self.k
        self.responsibilities = np.zeros((N, self.k))
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            # ========================================================
            # 1. E-Step (Expectation): Calcular Responsabilidades (Gamma)
            # Avalia a probabilidade a posteriori da variavel latente Z
            # ========================================================
            for j in range(self.k):
                self.responsibilities[:, j] = self.pi[j] * self._pdf(X, self.mu[j], self.sigma2[j])
            
            # Log-Likelihood para acompanhar convergência
            total_prob = np.sum(self.responsibilities, axis=1)
            log_likelihood = np.sum(np.log(total_prob + 1e-12))
            self.log_likelihoods.append(log_likelihood)
            
            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"EM convergiu na iteração {iteration}")
                break
            log_likelihood_old = log_likelihood
            
            # Normalizar responsabilidades
            self.responsibilities = self.responsibilities / total_prob[:, np.newaxis]
            
            # ========================================================
            # 2. M-Step (Maximization): Atualizar Parâmetros
            # Maximiza a Esperança da Log-Verossimilhança Completa
            # ========================================================
            Nj = np.sum(self.responsibilities, axis=0) # Número efetivo de observacoes no cluster
            
            for j in range(self.k):
                # Atualiza pesos (Priori)
                self.pi[j] = Nj[j] / N
                
                # Atualiza médias
                self.mu[j] = np.sum(self.responsibilities[:, j] * X) / Nj[j]
                
                # Atualiza variâncias
                self.sigma2[j] = np.sum(self.responsibilities[:, j] * ((X - self.mu[j])**2)) / Nj[j]
                
    def predict(self, X):
        """Retorna o regime mais provável (Hard Clustering)"""
        resp = np.zeros((len(X), self.k))
        for j in range(self.k):
            resp[:, j] = self.pi[j] * self._pdf(X, self.mu[j], self.sigma2[j])
        return np.argmax(resp, axis=1)
    
    def predict_proba(self, X):
        """Retorna a probabilidade pertença a cada regime (Soft Clustering)"""
        resp = np.zeros((len(X), self.k))
        for j in range(self.k):
            resp[:, j] = self.pi[j] * self._pdf(X, self.mu[j], self.sigma2[j])
        return resp / np.sum(resp, axis=1)[:, np.newaxis]

def generate_financial_data(T=1000):
    """
    Gera uma série temporal simulada da bolsa de valores com dois regimes:
    Regime 0 (Bull Market): Deriva positiva, baixa volatilidade.
    Regime 1 (Bear Market): Deriva negativa, alta volatilidade.
    """
    np.random.seed(42)
    prices = [100.0]
    returns = []
    regimes_true = []
    
    current_regime = 0
    # Matriz de transição de Markov P(i->j)
    # Probabilidade de manter o regime atual é alta
    transition_matrix = np.array([
        [0.98, 0.02], # Do Regime 0 (Bull)
        [0.05, 0.95]  # Do Regime 1 (Bear)
    ])
    
    mu_regimes = [0.001, -0.002]       # Média diária (Bull, Bear)
    sigma_regimes = [0.008, 0.025]     # Volatilidade diária (Bull, Bear)
    
    for t in range(T):
        regimes_true.append(current_regime)
        
        # Gerar o retorno do dia baseado no regime
        daily_return = np.random.normal(mu_regimes[current_regime], sigma_regimes[current_regime])
        returns.append(daily_return)
        
        # Preço do dia seguinte
        prices.append(prices[-1] * (1 + daily_return))
        
        # Transição de Markov para o próximo dia
        if np.random.rand() > transition_matrix[current_regime, current_regime]:
            current_regime = 1 - current_regime # Troca o regime (0 -> 1 ou 1 -> 0)
            
    return np.array(prices[:-1]), np.array(returns), np.array(regimes_true)

def analyze_market_regimes():
    # 1. Simulação dos dados de mercado
    T = 1500
    prices, returns, true_regimes = generate_financial_data(T)
    print("Dados da bolsa gerados com sucesso (1500 pregões).")
    
    # 2. Modelo de Mistura (Aprendizado Não Supervisionado)
    # O modelo não sabe quais são os regimes verdadeiros, ele aprenderá via EM
    print("Ajustando Modelo de Misturas Gaussianas (EM Algorithm)...")
    gmm = GaussianMixture1D_EM(k=2, max_iter=200, tol=1e-6)
    gmm.fit(returns)
    
    # Inferência dos regimes descobertos
    predicted_regimes = gmm.predict(returns)
    regime_probs = gmm.predict_proba(returns)
    
    # Corrigir simetria de rótulos (O cluster com menor média / alta variância é o BEAR)
    if gmm.sigma2[0] > gmm.sigma2[1]:
        # Trocar os rótulos 0 e 1 para que Regime 1 = Alta Volatilidade (Bear)
        predicted_regimes = 1 - predicted_regimes
        regime_probs = regime_probs[:, [1, 0]]
        mu_bull, mu_bear = gmm.mu[1], gmm.mu[0]
        var_bull, var_bear = gmm.sigma2[1], gmm.sigma2[0]
    else:
        mu_bull, mu_bear = gmm.mu[0], gmm.mu[1]
        var_bull, var_bear = gmm.sigma2[0], gmm.sigma2[1]

    print("\n--- Parâmetros Encontrados pelo Algoritmo EM ---")
    print(f"Regime Bull (Alta)  - Média: {mu_bull*100:.3f}% | Volatilidade: {np.sqrt(var_bull)*100:.3f}%")
    print(f"Regime Bear (Baixa) - Média: {mu_bear*100:.3f}% | Volatilidade: {np.sqrt(var_bear)*100:.3f}%")
    
    # 3. Visualização (Padrão Livro Aprendizado de Máquina)
    os.makedirs("figuras", exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=False, 
                                        gridspec_kw={'height_ratios': [2, 1, 1.5]})
    
    # Subplot 1: Dinâmica de Preços e Regimes Detectados
    ax1.set_title("Preço do Ativo na Bolsa e Regimes Detectados (GMM-EM)", fontsize=13)
    ax1.plot(prices, color='black', alpha=0.8, lw=1.5, label='Preço ($)')
    
    # Destacar o Regime Oculto (Soft Clustering probabilidade)
    # Cobre o fundo de vermelho onde o mercado é Bear (alta volatilidade)
    ax1.fill_between(range(T), 0, max(prices)*1.1, where=(predicted_regimes==1), 
                     color=VERMELHO, alpha=0.2, transform=ax1.get_xaxis_transform(), 
                     label='Regime Inferido: Bear Market')
    
    ax1.set_ylabel("Valor da Carteira ($)")
    ax1.legend(loc='upper left')
    
    # Subplot 2: Retornos Diários
    ax2.set_title("Série Estocástica de Retornos Diários", fontsize=11)
    ax2.plot(returns, color='gray', alpha=0.7, lw=1)
    ax2.set_ylabel("Retorno")
    
    # Subplot 3: Distribuição dos Retornos
    ax3.set_title("Identidade das Distribuições Inferidas (Misturas)", fontsize=11)
    x_axis = np.linspace(min(returns), max(returns), 500)
    
    ax3.hist(returns, bins=60, density=True, color='lightgrey', edgecolor='black', alpha=0.6, label='Retornos Empíricos')
    
    pdf_bull = (1.0 / np.sqrt(2 * np.pi * var_bull)) * np.exp(-0.5 * ((x_axis - mu_bull)**2) / var_bull)
    pdf_bear = (1.0 / np.sqrt(2 * np.pi * var_bear)) * np.exp(-0.5 * ((x_axis - mu_bear)**2) / var_bear)
    
    ax3.plot(x_axis, pdf_bull * gmm.pi[0 if mu_bull == gmm.mu[0] else 1], color=AZUL, lw=2.5, label='GMM: Componente Bull')
    ax3.plot(x_axis, pdf_bear * gmm.pi[1 if mu_bear == gmm.mu[1] else 0], color=VERMELHO, lw=2.5, label='GMM: Componente Bear')
    
    ax3.set_xlabel("Retorno Diário")
    ax3.set_ylabel("Densidade")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("figuras/regimes_financeiros_gmm.png", dpi=300)
    plt.close()
    
    print("\nVisualização salva com sucesso em 'figuras/regimes_financeiros_gmm.png'.")
    print("O script aplica os fundamentos do Capítulo 27 sobre dados financeiros latentes.")

if __name__ == "__main__":
    analyze_market_regimes()
