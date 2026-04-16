"""
Regressão Logística Regularizada com L1 (Lasso Logistic)
Baseado no Cap. 87 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo de Proximal Gradient Descent do zero para 
resolver o problema de otimização da logística com penalidade L1, 
promovendo a esparsidade e seleção automática de variáveis para análise de crédito.
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

class L1LogisticScratch:
    """Implementaçao manual de Logística L1 via Proximal Gradient Descent."""
    
    def __init__(self, lmbda=0.1, lr=0.01, max_iter=1000, tol=1e-6):
        self.lmbda = lmbda
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.intercept = 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def _proximal_operator(self, w, threshold):
        """Operador de Soft-Thresholding para a penalidade L1."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        self.intercept = 0
        
        print(f"Treinando Logística L1 (lambda={self.lmbda})...")
        
        for i in range(self.max_iter):
            beta_old = self.beta.copy()
            
            # 1. Gradiente da Log-Verossimilhança (perda logística)
            z = X @ self.beta + self.intercept
            p = self._sigmoid(z)
            
            error = p - y
            grad_beta = (X.T @ error) / n_samples
            grad_intercept = np.mean(error)
            
            # 2. Passo de Gradiente para o intercepto (não penalizado)
            self.intercept -= self.lr * grad_intercept
            
            # 3. Passo Proximal para os coeficientes beta
            # Descida de gradiente simples + Projeção L1
            step_beta = self.beta - self.lr * grad_beta
            self.beta = self._proximal_operator(step_beta, self.lr * self.lmbda)
            
            # Verificação de convergência
            if np.linalg.norm(self.beta - beta_old) < self.tol:
                print(f"Convergência atingida na iteração {i}")
                break
                
        return self

    def predict_proba(self, X):
        z = X @ self.beta + self.intercept
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# =============================================================================
# EXPERIMENTO: CREDIT SCORING E SELEÇÃO DE VARIÁVEIS
# =============================================================================

def run_credit_experiment():
    np.random.seed(42)
    N = 300
    p = 20 # 20 variáveis financeiras
    
    # Gerando dados: Apenas 4 variáveis são realmente preditivas de default
    X = np.random.randn(N, p)
    # Variáveis 0, 1, 2, 3 têm influência real
    true_beta = np.zeros(p)
    true_beta[:4] = [1.5, -2.0, 1.0, -1.2]
    
    z = X @ true_beta + np.random.normal(0, 0.5, N)
    p_true = 1 / (1 + np.exp(-z))
    y = (p_true > 0.5).astype(int)
    
    # Testar diferentes lambdas para ver o caminho de regularização
    lambdas = np.logspace(-3, 1, 20)
    coef_path = []
    
    for l in lambdas:
        model = L1LogisticScratch(lmbda=l, lr=0.1, max_iter=2000).fit(X, y)
        coef_path.append(model.beta)
    
    coef_path = np.array(coef_path)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for j in range(p):
        plt.plot(lambdas, coef_path[:, j], label=f'Var {j}' if j < 4 else "")
    
    plt.xscale('log')
    plt.xlabel('Força da Regularização (Lambda)')
    plt.ylabel('Valor do Coeficiente')
    plt.title("Lasso Logistic: Caminho de Regularização para Análise de Crédito\n(Destaque para variáveis 0, 1, 2 e 3)")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("figuras/credit_l1_path.png", dpi=300)
    plt.close()
    
    # Modelo Final com Lambda moderado
    final_model = L1LogisticScratch(lmbda=0.05, lr=0.1, max_iter=2000).fit(X, y)
    n_nonzero = np.sum(np.abs(final_model.beta) > 1e-4)
    
    print(f"\nResultados:")
    print(f"Variáveis selecionadas (não-zero): {n_nonzero} de {p}")
    print(f"Coeficientes das 4 principais: {final_model.beta[:4]}")
    print("Figura salva em 'figuras/credit_l1_path.png'")

if __name__ == "__main__":
    run_credit_experiment()
