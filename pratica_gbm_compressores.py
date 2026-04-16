"""
Gradient Boosting para Regressão (L2-Boost)
Baseado no Cap. 51 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo de Gradient Boosting do zero para 
minimização do erro quadrático, utilizando Decision Stumps como aprendizes fracos.
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

class DecisionStump:
    """Árvore de decisão de profundidade 1 para regressão."""
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_val = None
        self.right_val = None
        self.improvement = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_mse = np.inf
        
        # Busca exaustiva pelo melhor split
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                l_val = np.mean(y[left_mask])
                r_val = np.mean(y[right_mask])
                
                mse = np.sum((y[left_mask] - l_val)**2) + np.sum((y[right_mask] - r_val)**2)
                
                if mse < best_mse:
                    best_mse = mse
                    self.feature_idx = feat
                    self.threshold = t
                    self.left_val = l_val
                    self.right_val = r_val
        
        self.improvement = (np.var(y) * n_samples) - best_mse
        return self

    def predict(self, X):
        mask = X[:, self.feature_idx] <= self.threshold
        return np.where(mask, self.left_val, self.right_val)

class GradientBoostingScratch:
    """Implementaçao manual de Gradient Boosting Regressor (L2-Boost)."""
    
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.estimators = []
        self.init_val = None
        self.feature_importances_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        
        # 1. Inicializar com a média (Minimiza L2 inicial)
        self.init_val = np.mean(y)
        f_m = np.full(n_samples, self.init_val)
        
        print(f"Iniciando Boosting com {self.n_estimators} stumps...")
        
        for m in range(self.n_estimators):
            # 2. Calcular resíduos (Gradiente negativo da perda L2)
            residuals = y - f_m
            
            # 3. Ajustar um stump aos resíduos
            stump = DecisionStump().fit(X, residuals)
            
            # 4. Atualizar o modelo com Shrinkage
            f_m += self.lr * stump.predict(X)
            
            self.estimators.append(stump)
            self.feature_importances_[stump.feature_idx] += stump.improvement
            
        # Normalizar importâncias
        self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_val)
        for estimator in self.estimators:
            y_pred += self.lr * estimator.predict(X)
        return y_pred

# =============================================================================
# EXPERIMENTO: MANUTENÇÃO PREDITIVA DE COMPRESSORES
# =============================================================================

def run_gbm_experiment():
    np.random.seed(42)
    N = 200
    # Sensores: X0: Vibração (RMS), X1: Temperatura de Descarga, X2: Corrente Elétrica
    X = np.random.rand(N, 3) * 10
    
    # Índice de Saúde (Y): Dependência não-linear complexa
    # Saúde cai se Vibração > 7 E Temperatura > 8 (Interação)
    def health_index(X_val):
        vib, temp, curr = X_val[:, 0], X_val[:, 1], X_val[:, 2]
        base = 100 - (0.5 * vib**2) - (0.3 * temp**2)
        # Interação crítica
        base -= 2.0 * np.maximum(0, vib - 7) * np.maximum(0, temp - 8)
        return base
    
    y = health_index(X) + np.random.normal(0, 2, N)
    
    # Ajustar GBM
    model = GradientBoostingScratch(n_estimators=100, learning_rate=0.1).fit(X, y)
    y_pred = model.predict(X)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Real vs Estimado
    axes[0].scatter(y, y_pred, color='#1f77b4', alpha=0.6)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title("GBM: Índice de Saúde do Compressor\n(Real vs Estimado)")
    axes[0].set_xlabel("Saúde Real")
    axes[0].set_ylabel("Saúde Predita")
    
    # 2. Importância das Características
    sensors = ['Vibração', 'Temperatura', 'Corrente']
    axes[1].bar(sensors, model.feature_importances_, color='#2ca02c')
    axes[1].set_title("Importância dos Sensores\n(Contribuição para o Diagnóstico)")
    axes[1].set_ylabel("Importância Relativa")
    
    plt.tight_layout()
    plt.savefig("figuras/gbm_compressor_results.png", dpi=300)
    plt.close()
    
    print(f"\nResultados:")
    print(f"Erro Quadrático Médio (MSE): {np.mean((y - y_pred)**2):.4f}")
    print(f"Importâncias: {dict(zip(sensors, model.feature_importances_))}")
    print("Figura salva em 'figuras/gbm_compressor_results.png'")

if __name__ == "__main__":
    run_gbm_experiment()
