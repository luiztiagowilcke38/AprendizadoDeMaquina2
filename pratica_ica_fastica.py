"""
Análise de Componentes Independentes (ICA) via FastICA
Baseado no Cap. 35 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo FastICA do zero para resolver o problema de 
separação cega de fontes (Blind Source Separation), maximizando a 
não-gaussianidade das projeções.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

def g(x):
    """Função de contraste (log-cosh)."""
    return np.tanh(x)

def g_prime(x):
    """Derivada da função de contraste."""
    return 1 - np.tanh(x)**2

class FastICAScratch:
    """Implementação manual do algoritmo FastICA (Fixed-Point)."""
    
    def __init__(self, n_components=2, max_iter=200, tol=1e-4):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None
        
    def _whiten(self, X):
        """Branqueamento (Whitening) via SVD."""
        # Centralizar
        X_centered = X - np.mean(X, axis=0)
        # SVD: X = U S V'
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # X_white = X_centered * V * S^-1 = U
        # Na prática, usamos: X_white = X_centered @ V @ diag(1/S) * sqrt(N-1)
        X_white = np.sqrt(X.shape[0] - 1) * U
        # Matriz de branqueamento para uso futuro se necessário
        # K = V @ diag(1/S) * sqrt(N-1)
        return X_white

    def fit_transform(self, X):
        # 1. Branqueamento
        X_w = self._whiten(X)
        n_samples, n_features = X_w.shape
        
        # 2. Inicializar matriz de desmistura W
        W = np.random.rand(self.k, n_features)
        
        print(f"Iniciando FastICA para {self.k} componentes...")
        
        for i in range(self.k):
            w = W[i, :].copy()
            w /= np.linalg.norm(w)
            
            for it in range(self.max_iter):
                # Regra de atualização fixa (E[x * g(w'x)] - E[g'(w'x)] * w)
                w_new = (X_w.T @ g(X_w @ w)) / n_samples - np.mean(g_prime(X_w @ w)) * w
                
                # Ortogonalização de Gram-Schmidt (decorrelação com componentes anteriores)
                if i > 0:
                    w_new -= (w_new @ W[:i, :].T) @ W[:i, :]
                
                w_new /= np.linalg.norm(w_new)
                
                # Convergência
                if np.abs(np.abs(w @ w_new) - 1) < self.tol:
                    break
                w = w_new
            
            W[i, :] = w
            
        self.W = W
        return X_w @ W.T

# =============================================================================
# EXPERIMENTO "COSMIC SEPARATION": PULSAR VS. RUÍDO DE FUNDO
# =============================================================================

def run_ica_experiment():
    np.random.seed(42)
    n_samples = 2000
    t = np.linspace(0, 10, n_samples)
    
    # 1. Fontes Independentes (S)
    # Fonte 1: Sinal de Pulsar (Onda quadrada/pulso)
    s1 = np.where(np.sin(2 * np.pi * 2 * t) > 0.9, 1.0, 0.0)
    # Fonte 2: Ruído Estocástico (Econômico ou Cósmico - ex: Dente de serra com ruído)
    s2 = (t % 1) - 0.5 
    
    S = np.stack([s1, s2], axis=1)
    
    # 2. Mistura (A)
    # Imaginamos dois sensores (ex: dois radiotelescópios ou dois indicadores econômicos)
    A = np.array([[0.6, 0.4], [0.5, 0.8]])
    X = S @ A.T
    
    # 3. Aplicar ICA
    ica = FastICAScratch(n_components=2)
    S_rec = ica.fit_transform(X)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Fontes Originais
    axes[0].plot(t, s1, color='#2ca02c', label='Fonte 1: Pulsar')
    axes[0].plot(t, s2, color='#9467bd', alpha=0.7, label='Fonte 2: Tendência')
    axes[0].set_title("Fontes Independentes Originais (S)")
    axes[0].legend(loc='upper right')
    
    # Sinais Misturados (O que os sensores captam)
    axes[1].plot(t, X[:, 0], color='gray', alpha=0.8, label='Sensor 1')
    axes[1].plot(t, X[:, 1], color='black', alpha=0.5, label='Sensor 2')
    axes[1].set_title("Sinais Observados / Misturados (X = AS)")
    axes[1].legend(loc='upper right')
    
    # Sinais Recuperados por ICA
    # Nota: ICA recupera fontes com escala e sinal arbitrários.
    axes[2].plot(t, S_rec[:, 0], color='#1f77b4', label='ICA Comp. 1')
    axes[2].plot(t, S_rec[:, 1], color='#ff7f0e', label='ICA Comp. 2')
    axes[2].set_title("Fontes Recuperadas via FastICA (S' = WX)")
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("figuras/ica_fastica_results.png", dpi=300)
    plt.close()
    
    print(f"\nResultados:")
    print(f"Sinais recuperados e figura salva em 'figuras/ica_fastica_results.png'")

if __name__ == "__main__":
    run_ica_experiment()
