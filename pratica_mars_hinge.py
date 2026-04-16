"""
Multivariate Adaptive Regression Splines (MARS) via Forward Stepwise
Baseado no Cap. 38 do livro 'Aprendizado de Máquina'.

Este script implementa a fase 'Forward' do algoritmo MARS do zero, 
construindo um modelo adaptativo de splines lineares para capturar 
não-linearidades e interações em dados econômicos.
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

class BasisFunction:
    """Classe base para funções de base do MARS (Constant ou Hinge)."""
    def __init__(self, feature_idx=None, knot=None, side=None, parent=None):
        self.feature_idx = feature_idx
        self.knot = knot
        self.side = side  # +1 para (x-t)+, -1 para (t-x)+
        self.parent = parent # Referência à função de base pai (para interações)
        
        # Grau de interação: grau do pai + 1 (ConstantBasis tem grau 0)
        self.degree = 0 if parent is None else parent.degree + 1

    def apply(self, X):
        """Aplica a função de base aos dados X."""
        # Se for constante (feature_idx é None)
        if self.feature_idx is None:
            return np.ones(X.shape[0])
        
        # Se for hinge: h_m(x) * side * (x_j - t)_+
        hinge = self.side * (X[:, self.feature_idx] - self.knot)
        val = np.maximum(0, hinge)
        
        if self.parent:
            return self.parent.apply(X) * val
        return val

class MARSScratch:
    """Implementaçao robusta do MARS (Apenas Forward Pass)."""
    
    def __init__(self, max_terms=15, max_degree=2):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.basis_functions = [BasisFunction()] # Inicia com termo constante (grau 0)
        self.beta = None
        
    def _get_basis_matrix(self, X):
        """Constrói a matriz H de bases atual."""
        N = X.shape[0]
        M = len(self.basis_functions)
        H = np.zeros((N, M))
        for i, bf in enumerate(self.basis_functions):
            H[:, i] = bf.apply(X)
        return H

    def fit(self, X, y):
        N, p = X.shape
        y = y.reshape(-1, 1)
        
        print(f"Iniciando Forward Stepwise MARS ({self.max_terms} termos max)...")
        
        # Loop para adicionar pares de bases até atingir max_terms
        while len(self.basis_functions) + 2 <= self.max_terms:
            best_rss = np.inf
            best_pair = None
            
            H_current = self._get_basis_matrix(X)
            
            # Busca exaustiva: O(N * p * M)
            # Nota: Para cada função de base existente m, tentamos interações com cada feature j
            for m_idx, parent_bf in enumerate(self.basis_functions):
                # Controle de grau de interação
                if parent_bf.degree >= self.max_degree:
                    continue
                
                for j in range(p):
                    # Candidatos a nó (knots): usamos todos os valores únicos observados da feature j
                    # (Em datasets grandes, recomenda-se usar quantis para eficiência)
                    unique_values = np.unique(X[:, j])
                    for t in unique_values:
                        # Criar potenciais novas funções de base (o par refletido)
                        bf_plus = BasisFunction(feature_idx=j, knot=t, side=1, parent=parent_bf)
                        bf_minus = BasisFunction(feature_idx=j, knot=t, side=-1, parent=parent_bf)
                        
                        # Vetores das novas bases
                        h_plus = bf_plus.apply(X)
                        h_minus = bf_minus.apply(X)
                        
                        # Matriz candidata
                        H_new = np.column_stack([H_current, h_plus, h_minus])
                        
                        # Resolver OLS para avaliar o RSS
                        # complexidade O(N * M^2)
                        beta_tmp, residues, _, _ = np.linalg.lstsq(H_new, y, rcond=None)
                        
                        # Cálculo manual do RSS se residues vier vazio (sistema subdeterminado)
                        current_rss = residues[0] if residues.size > 0 else np.sum((y - H_new @ beta_tmp)**2)
                        
                        if current_rss < best_rss:
                            best_rss = current_rss
                            best_pair = (bf_plus, bf_minus)
            
            if best_pair:
                self.basis_functions.extend([best_pair[0], best_pair[1]])
                # print(f"Adicionados termos com grau {best_pair[0].degree}. Total: {len(self.basis_functions)}")
            else:
                break
                
        # Ajuste final do modelo
        H_final = self._get_basis_matrix(X)
        self.beta, _, _, _ = np.linalg.lstsq(H_final, y, rcond=None)
        print(f"Modelo finalizado com {len(self.basis_functions)} funções de base.")
        return self

    def predict(self, X):
        H = self._get_basis_matrix(X)
        return H @ self.beta

# =============================================================================
# EXPERIMENTO: DINÂMICA DE DESEMPREGO (CURVA DE PHILLIPS)
# =============================================================================

def run_mars_experiment():
    np.random.seed(42)
    N = 100
    
    # Geração de dados consistente (SEM np.sort independente por coluna)
    # X1: Crescimento do PIB (GDP), X2: Inflação
    X = np.random.uniform(-2, 5, (N, 2))
    
    # Modelo Real: Curva de Phillips kincada + Interação entre PIB e Inflação
    def true_model(X_val):
        gdp = X_val[:, 0]
        inf = X_val[:, 1]
        # Base: Desemprego cai com PIB alto, mas tem um 'chão' (non-linear)
        y_val = 6 - 0.8 * np.maximum(0, gdp - 1) + 2 * np.maximum(0, 1 - gdp)
        # Interação: Se inflação > 3, o efeito do PIB no desemprego muda
        y_val += 0.5 * np.maximum(0, inf - 3) * np.maximum(0, gdp)
        return y_val
    
    y_true = true_model(X)
    y_noisy = y_true + 0.3 * np.random.randn(N)
    
    # Ajustar MARS
    model = MARSScratch(max_terms=10, max_degree=2).fit(X, y_noisy)
    
    # Para visualização, criamos um grid ou ordenamos CORRETAMENTE
    X_test = np.zeros((100, 2))
    X_test[:, 0] = np.linspace(-2, 5, 100) # Variando PIB
    X_test[:, 1] = 2.0 # Inflação constante para o corte 1D
    
    y_test_true = true_model(X_test)
    y_test_pred = model.predict(X_test)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(X[:, 0], y_noisy, color='gray', alpha=0.3, label='Dados Observados (PIB vs Desemprego)')
    plt.plot(X_test[:, 0], y_test_true, color='black', lw=2, label='Modelo Real (Phillips Curve + Interações)')
    plt.plot(X_test[:, 0], y_test_pred, color='#d62728', lw=3, label='MARS Estimado (Forward Pass)')
    
    plt.title("MARS Refatorado: Capturando Dinâmicas Não-Lineares de Emprego")
    plt.xlabel("Crescimento do PIB (%)")
    plt.ylabel("Taxa de Desemprego (%)")
    plt.legend()
    plt.savefig("figuras/mars_unemployment_results.png", dpi=300)
    plt.close()
    
    print(f"\nResultados:")
    print(f"Termos de base aprendidos: {len(model.basis_functions)}")
    print("Figura de validação salva em 'figuras/mars_unemployment_results.png'")

if __name__ == "__main__":
    run_mars_experiment()
