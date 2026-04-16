"""
Modelos Aditivos Generalizados (GAM) via Backfitting e Cubic Splines
Baseado no Cap. 37 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo de Backfitting do zero para resolver modelos da forma:
y = alpha + f1(x1) + f2(x2) + ... + fp(xp) + erro
Sem depender de bibliotecas externas para o motor de ajuste.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas (Seguindo o padrão do livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'
ROXO = '#9467bd'

# 1. COMPONENTE: NATURAL CUBIC SPLINES (Conforme Cap. 37 do livro)

class NaturalCubicSpline:
    """Implementação manual de Cubic Splines de Suavização."""
    def __init__(self, smoothing_param=0.5):
        self.lam = smoothing_param  # Parâmetro de regularização (lambda)
        
    def _create_basis(self, x, knots):
        """Cria matriz de base para Splines Cúbicas."""
        N = len(x)
        K = len(knots)
        X_base = np.zeros((N, K))
        
        # d_k(x) = ( (x-knots[k])_+^3 - (x-knots[K-1])_+^3 ) / (knots[K-1] - knots[k])
        def d(x, k, knots):
            K_last = knots[-1]
            k_val = knots[k]
            num = (np.maximum(0, x - k_val)**3 - np.maximum(0, x - K_last)**3)
            den = (K_last - k_val)
            return num / den

        X_base[:, 0] = 1
        X_base[:, 1] = x
        for k in range(K - 2):
            X_base[:, k+2] = d(x, k, knots) - d(x, K-2, knots)
            
        return X_base

    def fit_predict(self, x, y):
        """Ajusta e prediz para um conjunto de dados 1D."""
        # Ordenar para garantir knots uniformes
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        
        # Knots nos percentis (robusto)
        knots = np.percentile(x_sorted, np.linspace(0, 100, 10))
        
        # Matriz de Base H
        H = self._create_basis(x_sorted, knots)
        
        # Penalidade de curvatura (Proxy via Ridge Regression na base no espaco H)
        # Em uma implementação completa de ESL, usaríamos a matriz Omega de segundas derivadas.
        # Aqui usamos penalidade L2 com pesos para simplificar o motor de backfitting.
        I = np.eye(H.shape[1])
        I[0,0] = 0 # Nao penaliza intercepto
        I[1,1] = 0 # Nao penaliza termo linear
        
        # Solução via Mínimos Quadrados Penalizados (Equações Normais)
        # theta = (H'H + lam*I)^-1 H'y
        # Note: Implementação baseada na penalização de curvatura do livro.
        try:
            theta = np.linalg.solve(H.T @ H + self.lam * I, H.T @ y_sorted)
            y_hat_sorted = H @ theta
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(H.T @ H + self.lam * I, H.T @ y_sorted, rcond=None)[0]
            y_hat_sorted = H @ theta
            
        # Reordenar para a ordem original
        y_hat = np.zeros_like(y)
        y_hat[idx] = y_hat_sorted
        return y_hat - np.mean(y_hat) # Remove média para centralização no GAM

# 2. MOTOR: ALGORITMO DE BACKFITTING (Conforme Cap. 37 do livro)

class BackfittingGAM:
    """Implementação do Algoritmo de Backfitting do livro."""
    def __init__(self, n_features, lam=0.5, max_iter=20, tol=1e-4):
        self.n_features = n_features
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.f_hat = None # Armazena as funções estimadas f_j
        self.alpha = 0    # Intercepto global
        self.splines = [NaturalCubicSpline(smoothing_param=lam) for _ in range(n_features)]

    def fit(self, X, y):
        N = X.shape[0]
        self.alpha = np.mean(y)
        self.f_hat = np.zeros((N, self.n_features))
        
        prev_f = self.f_hat.copy()
        
        print(f"Iniciando Backfitting para {self.n_features} variáveis...")
        for it in range(self.max_iter):
            for j in range(self.n_features):
                # Calcular resíduo parcial: r = y - alpha - sum(f_k for k!=j)
                other_f_sum = np.sum(self.f_hat, axis=1) - self.f_hat[:, j]
                partial_resid = y - self.alpha - other_f_sum
                
                # Ajustar suavizador aos resíduos parciais
                self.f_hat[:, j] = self.splines[j].fit_predict(X[:, j], partial_resid)
            
            # Verificação de convergência
            delta = np.linalg.norm(self.f_hat - prev_f) / (np.linalg.norm(prev_f) + 1e-9)
            if delta < self.tol:
                print(f"Convergência atingida na iteração {it+1} (delta={delta:.6f})")
                break
            prev_f = self.f_hat.copy()
            if it == self.max_iter - 1:
                print("Aviso: Backfitting atingiu o máximo de iterações.")

    def predict(self, X):
        return self.alpha + np.sum(self.f_hat, axis=1)

# =============================================================================
# 3. EXPERIMENTO E VISUALIZAÇÃO
# =============================================================================

def run_gam_experiment():
    np.random.seed(42)
    N = 400
    
    # Gerando dados multivariados com não-linearidades distintas
    X1 = np.random.uniform(-3, 3, N)
    X2 = np.random.uniform(-3, 3, N)
    X3 = np.random.uniform(-3, 3, N)
    
    # f1: Senoidal, f2: Quadrática, f3: Linear
    y_true_f1 = np.sin(X1 * 1.5) * 2
    y_true_f2 = (X2**2 - 3) * 0.5
    y_true_f3 = X3 * 0.8
    
    y = 5.0 + y_true_f1 + y_true_f2 + y_true_f3 + np.random.normal(0, 0.5, N)
    X = np.stack([X1, X2, X3], axis=1)
    
    # Ajustar GAM
    gam = BackfittingGAM(n_features=3, lam=5.0)
    gam.fit(X, y)
    
    y_pred = gam.predict(X)
    
    # VISUALIZAÇÃO DAS FUNÇÕES PARCIAIS (INTERPRETABILIDADE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [r"$f_1(X_1) = \sin(1.5X_1)$", r"$f_2(X_2) = 0.5(X_2^2-3)$", r"$f_3(X_3) = 0.8X_3$"]
    x_labels = ["X1", "X2", "X3"]
    
    for j in range(3):
        idx = np.argsort(X[:, j])
        axes[j].scatter(X[:, j], y - gam.alpha - (np.sum(gam.f_hat, axis=1) - gam.f_hat[:, j]), 
                        alpha=0.2, color='gray', s=10, label='Resíduo Parcial')
        axes[j].plot(X[idx, j], gam.f_hat[idx, j], color=AZUL, lw=3, label='Estimativa GAM')
        
        # Plotar verdade (ajustada para media zero)
        actual_true = [y_true_f1, y_true_f2, y_true_f3][j]
        axes[j].plot(X[idx, j], actual_true[idx] - np.mean(actual_true), '--', color=LARANJA, label='Verdadeiro')
        
        axes[j].set_title(titles[j])
        axes[j].set_xlabel(x_labels[j])
        axes[j].legend()
    
    plt.suptitle("Modelos Aditivos Generalizados (GAM): Recuperação de Funções via Backfitting", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs("figuras", exist_ok=True)
    plt.savefig("figuras/gam_backfitting_results.png", dpi=300)
    plt.close()
    
    # Comparação de Erro
    mse_gam = np.mean((y - y_pred)**2)
    print(f"\nResultados:")
    print(f"Erro Quadrático Médio (GAM): {mse_gam:.4f}")
    
    # Comparação com Regressão Linear Simples
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X, y)
    mse_lr = np.mean((y - lr.predict(X))**2)
    print(f"Erro Quadrático Médio (Regr. Linear): {mse_lr:.4f}")
    print(f"Melhoria do GAM: {((mse_lr - mse_gam)/mse_lr)*100:.1f}%")

if __name__ == "__main__":
    run_gam_experiment()
    print("\nScript finalizado com sucesso! Figura salva em 'figuras/gam_backfitting_results.png'")
