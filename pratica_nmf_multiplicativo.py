"""
Fatoração de Matrizes Não-Negativas (NMF) via Atualizações Multiplicativas
Baseado no Cap. 69 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo de NMF do zero para resolver a decomposição V ≈ WH,
onde todas as matrizes são estritamente não-negativas.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": False
})

class MultiplicativeNMF:
    """Implementação manual de NMF com regras de atualização multiplicativa."""
    
    def __init__(self, n_components=2, objective='frobenius', max_iter=200, tol=1e-5):
        self.k = n_components
        self.objective = objective
        self.max_iter = max_iter
        self.tol = tol
        self.W = None
        self.H = None
        self.history = []

    def _update_frobenius(self, V, W, H):
        """Atualização para o erro quadrático (Norma de Frobenius)."""
        # H <- H * (W.T @ V) / (W.T @ W @ H)
        numerator_h = W.T @ V
        denominator_h = W.T @ W @ H + 1e-9
        H *= (numerator_h / denominator_h)
        
        # W <- W * (V @ H.T) / (W @ H @ H.T)
        numerator_w = V @ H.T
        denominator_w = W @ H @ H.T + 1e-9
        W *= (numerator_w / denominator_w)
        
        return W, H

    def _update_kl(self, V, W, H):
        """Atualização para a divergência de Kullback-Leibler."""
        # H <- H * (W.T @ (V / (W @ H))) / col_sum(W)
        WH = W @ H + 1e-9
        H *= (W.T @ (V / WH)) / np.sum(W, axis=0)[:, None]
        
        # W <- W * ((V / (W @ H)) @ H.T) / row_sum(H)
        WH = W @ H + 1e-9
        W *= ((V / WH) @ H.T) / np.sum(H, axis=1)[None, :]
        
        return W, H

    def fit(self, V):
        N, M = V.shape
        # Inicialização positiva aleatória
        self.W = np.random.rand(N, self.k) + 0.1
        self.H = np.random.rand(self.k, M) + 0.1
        
        print(f"Iniciando NMF ({self.objective}) com k={self.k}...")
        
        for i in range(self.max_iter):
            if self.objective == 'frobenius':
                self.W, self.H = self._update_frobenius(V, self.W, self.H)
                error = np.linalg.norm(V - self.W @ self.H, 'fro')
            else: # kl
                self.W, self.H = self._update_kl(V, self.W, self.H)
                # Proxy de erro simples para KL
                WH = self.W @ self.H + 1e-9
                error = np.sum(V * np.log(V/WH + 1e-9) - V + WH)
            
            self.history.append(error)
            
            if i > 0 and abs(self.history[-2] - self.history[-1]) < self.tol:
                print(f"Convergência atingida na iteração {i+1}.")
                break
                
        return self.W, self.H

# =============================================================================
# EXPERIMENTO: APRENDIZADO DE BASES LOCAIS (BLOCKS WORLD)
# =============================================================================

def run_nmf_experiment():
    np.random.seed(42)
    
    # Criando uma matriz de "imagens" 10x10 planificadas (100 pixels)
    # Vamos criar 4 bases fundamentais (blocos 5x5 em diferentes quadrantes)
    size = 10
    n_features = size * size
    n_samples = 300
    
    bases_true = np.zeros((4, n_features))
    # Quadrante superior esquerdo
    bases_true[0, [i*size + j for i in range(5) for j in range(5)]] = 1
    # Quadrante superior direito
    bases_true[1, [i*size + j for i in range(5) for j in range(5, 10)]] = 1
    # Quadrante inferior esquerdo
    bases_true[2, [i*size + j for i in range(5, 10) for j in range(5)]] = 1
    # Quadrante inferior direito
    bases_true[3, [i*size + j for i in range(5, 10) for j in range(5, 10)]] = 1
    
    # Gerando dados como combinações positivas dessas bases
    H_true = np.random.exponential(scale=1.0, size=(n_samples, 4))
    V = H_true @ bases_true
    # Adicionando um pouco de ruído positivo
    V += np.random.uniform(0, 0.05, V.shape)
    
    # Ajustar NMF
    nmf = MultiplicativeNMF(n_components=4, objective='frobenius', max_iter=500)
    W, H = nmf.fit(V) # W (samples x k), H (k x features)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    # 1. Plot das Bases Aprendidas (as linhas de H representam as bases de imagem)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        axes[i].imshow(H[i].reshape(size, size), cmap='viridis')
        axes[i].set_title(f"Base {i+1}")
        axes[i].axis('off')
    
    plt.suptitle("Bases Aprendidas pelo NMF (Recuperação de Estrutura Local)", fontsize=14)
    plt.savefig("figuras/nmf_bases.png", dpi=300)
    plt.close()
    
    # 2. Curva de Erro
    plt.figure(figsize=(8, 4))
    plt.plot(nmf.history, color='#1f77b4', lw=2)
    plt.title("Convergência do NMF (Erro de Reconstrução)")
    plt.xlabel("Iteração")
    plt.ylabel("Erro Frobenius")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig("figuras/nmf_convergence.png", dpi=300)
    plt.close()
    
    print(f"\nResultados:")
    print(f"Erro final de reconstrução: {nmf.history[-1]:.4f}")
    print("Figuras salvas em 'figuras/nmf_bases.png' e 'figuras/nmf_convergence.png'")

if __name__ == "__main__":
    run_nmf_experiment()
