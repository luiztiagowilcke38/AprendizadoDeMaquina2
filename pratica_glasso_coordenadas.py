"""
Graphical Lasso (GLASSO) via Descida por Coordenadas de Bloco
Baseado no Cap. 73 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo GLASSO do zero para estimar matrizes de precisão 
esparsas (inversa da covariância). O algoritmo resolve o problema 
min [ tr(S*Theta) - log det(Theta) + lam*||Theta||_1 ] através de uma 
sequência de problemas de regressão Lasso.
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

def soft_threshold(x, lam):
    """Operador de limiarização suave (Soft-thresholding)."""
    return np.sign(x) * np.maximum(0, np.abs(x) - lam)

def lasso_coordinate_descent(X, y, lam, max_iter=100, tol=1e-4):
    """Solver progressivo de Lasso via descida por coordenadas."""
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(n_features):
            # Calcular resíduo parcial sem a característica j
            r = y - (X @ beta - X[:, j] * beta[j])
            # Coordenada ótima
            num = X[:, j].T @ r
            den = X[:, j].T @ X[:, j]
            beta[j] = soft_threshold(num, lam) / (den + 1e-9)
            
        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta

class GraphicalLassoScratch:
    """Implementação manual do Graphical Lasso baseada em Friedman et al. (2008)."""
    
    def __init__(self, lam=0.1, max_iter=20, tol=1e-4):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_ = None
        self.precision_ = None

    def fit(self, X):
        S = np.cov(X, rowvar=False)
        p = S.shape[0]
        
        # Inicializar covariância estimada como S + lam*I (garante positividade)
        W = S + self.lam * np.eye(p)
        Theta = np.linalg.inv(W)
        
        print(f"Iniciando GLASSO com lambda={self.lam}...")
        
        for it in range(self.max_iter):
            W_old = W.copy()
            
            for j in range(p):
                # Particionar W e S
                # w12: coluna j de W (excluindo elemento jj)
                # W11: matriz W excluindo linha/coluna j
                indices = np.delete(np.arange(p), j)
                W11 = W[np.ix_(indices, indices)]
                s12 = S[indices, j]
                
                # O problema do Glasso para uma coluna reduz-se a:
                # min beta' W11 beta s.t. ||beta||_1 <= s12
                # Que é equivalente a um Lasso de s12 contra W11
                # Resolvemos o problema dual equivalente:
                # beta = argmin { 1/2 * beta' * W11 * beta - beta' * s12 + lam * ||beta||_1 }
                # Usamos decomposição espectral para acelerar a regressão ou solver direto
                
                # Nota: W11 é a nossa matriz de 'dados' (gramiana) e s12 o alvo
                # Beta representa o vetor de coeficientes que relaciona a variável j com as outras
                beta = lasso_coordinate_descent_on_gram(W11, s12, self.lam)
                
                # Atualizar w12
                w12 = W11 @ beta
                W[indices, j] = w12
                W[j, indices] = w12
                
                # Atualizar Theta (Precisão) usando complemento de Schur
                theta_jj = 1 / (W[j, j] - w12.T @ beta + 1e-9)
                theta_12 = -theta_jj * beta
                Theta[j, j] = theta_jj
                Theta[np.ix_(indices, [j])] = theta_12.reshape(-1, 1)
                Theta[np.ix_([j], indices)] = theta_12.reshape(1, -1)

            if np.linalg.norm(W - W_old, ord='fro') < self.tol:
                print(f"Convergência atingida na iteração {it+1}.")
                break
                
        self.covariance_ = W
        self.precision_ = Theta
        return self

def lasso_coordinate_descent_on_gram(W11, s12, lam, max_iter=50):
    """Solver Lasso que opera diretamente na matriz gramiana W11 (mais rápido para Glasso)."""
    p = W11.shape[0]
    beta = np.zeros(p)
    
    for _ in range(max_iter):
        for j in range(p):
            # Gradiente: W11_j' * beta - s12_j
            # Onde W11_j é a coluna j de W11
            # s12_j é o j-ésimo elemento de s12
            gradient_step = s12[j] - (W11[j, :] @ beta - W11[j, j] * beta[j])
            beta[j] = soft_threshold(gradient_step, lam) / (W11[j, j] + 1e-9)
            
    return beta

# =============================================================================
# EXPERIMENTO: RECUPERAÇÃO DE ESTRUTURA DE GRAFO
# =============================================================================

def run_glasso_experiment():
    np.random.seed(42)
    p = 10  # Número de variáveis
    n = 100 # Número de amostras
    
    # Criar matriz de precisão esparsa (Grafo em anel/corrente)
    Theta_true = np.eye(p)
    for i in range(p - 1):
        Theta_true[i, i+1] = 0.5
        Theta_true[i+1, i] = 0.5
    
    # Garantir que seja definida positiva
    Theta_true += np.eye(p) * 0.1
    
    # Matriz de covariância verdadeira
    Sigma_true = np.linalg.inv(Theta_true)
    
    # Gerar dados
    X = np.random.multivariate_normal(np.zeros(p), Sigma_true, size=n)
    
    # Ajustar GLASSO
    model = GraphicalLassoScratch(lam=0.1)
    model.fit(X)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normalizar para visualização clara do padrão de esparsidade
    def normalize_matrix(M):
        M_abs = np.abs(M)
        return M_abs / (np.max(M_abs) + 1e-9)

    ax0 = axes[0].imshow(normalize_matrix(Theta_true), cmap='Blues', interpolation='nearest')
    axes[0].set_title("Matriz de Precisão Verdadeira\n(Estrutura de Anel)")
    fig.colorbar(ax0, ax=axes[0])
    
    ax1 = axes[1].imshow(normalize_matrix(model.precision_), cmap='Blues', interpolation='nearest')
    axes[1].set_title("Matriz de Precisão Estimada\n(Graphical Lasso)")
    fig.colorbar(ax1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("figuras/glasso_results.png", dpi=300)
    plt.close()
    
    print(f"\nResultados:")
    print(f"Norma de Frobenius do erro (Precisão): {np.linalg.norm(Theta_true - model.precision_):.4f}")
    print(f"Esparsidade obtida: {np.mean(np.abs(model.precision_) < 1e-4)*100:.1f}%")
    print("Figura salva em 'figuras/glasso_results.png'")

if __name__ == "__main__":
    run_glasso_experiment()
