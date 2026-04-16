"""
Análise de Sobrevivência e Modelo de Riscos Proporcionais de Cox
Baseado no Cap. 43 do livro 'Aprendizado de Máquina'.

Este script implementa o Modelo de Cox do zero, estimando os coeficientes via 
maximização da Verossimilhança Parcial usando o método de Newton-Raphson.
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

class CoxPHScratch:
    """Implementação manual do Modelo de Cox (Proportional Hazards)."""
    
    def __init__(self, max_iter=20, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        
    def _partial_log_likelihood(self, X, T, E, beta):
        """Calcula log-verossimilhança parcial, gradiente e hessiana."""
        N, p = X.shape
        # Ordenar por tempo de sobrevivência (descendente para facilitar R_i)
        idx = np.argsort(T)
        X, T, E = X[idx], T[idx], E[idx]
        
        log_lk = 0
        gradient = np.zeros(p)
        hessian = np.zeros((p, p))
        
        for i in range(N):
            if E[i] == 1: # Apenas se houve evento
                # Risk set R_i: todos os indivíduos j onde T_j >= T_i
                risk_indices = np.where(T >= T[i])[0]
                sum_exp = np.sum(np.exp(X[risk_indices] @ beta))
                
                # Log-Likelihood
                log_lk += (X[i] @ beta) - np.log(sum_exp + 1e-9)
                
                # Gradiente
                phi = np.exp(X[risk_indices] @ beta)
                weighted_avg_X = (phi[:, None] * X[risk_indices]).sum(axis=0) / sum_exp
                gradient += X[i] - weighted_avg_X
                
                # Hessiana (Calculando como variância ponderada de X)
                # V = sum(phi * (X - weighted_avg_X)^2) / sum_exp
                diff = X[risk_indices] - weighted_avg_X
                weighted_cov_X = (phi[:, None, None] * np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / sum_exp
                hessian -= weighted_cov_X
                
        return log_lk, gradient, hessian

    def fit(self, X, T, E):
        N, p = X.shape
        self.beta = np.zeros(p)
        
        print(f"Iniciando Newton-Raphson para Cox PH ({N} amostras)...")
        
        for it in range(self.max_iter):
            log_lk, gradient, hessian = self._partial_log_likelihood(X, T, E, self.beta)
            
            # Newton step: beta = beta - H^-1 * g
            try:
                update = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
                
            self.beta -= update
            
            if np.linalg.norm(update) < self.tol:
                print(f"Convergência atingida na iteração {it+1}.")
                break
        return self

    def predict_hazard_ratio(self, X):
        return np.exp(X @ self.beta)

    def baseline_survival(self, X, T, E):
        """Estimador de Breslow para a função de sobrevivência base S0(t)."""
        idx = np.argsort(T)
        X_s, T_s, E_s = X[idx], T[idx], E[idx]
        
        unique_times = np.unique(T_s[E_s == 1])
        h0 = []
        
        for t in unique_times:
            risk_indices = np.where(T_s >= t)[0]
            sum_exp = np.sum(np.exp(X_s[risk_indices] @ self.beta))
            # d_j / sum(exp(x'beta))
            # Onde d_j e o numero de eventos no tempo t
            d_j = np.sum((T_s == t) & (E_s == 1))
            h0.append(d_j / (sum_exp + 1e-9))
            
        # Funcao de Sobrevivencia: S(t) = exp(-sum(h0))
        H0 = np.cumsum(h0)
        S0 = np.exp(-H0)
        return unique_times, S0

# =============================================================================
# EXPERIMENTO: MEDICINA E FINANÇAS (SOBREVIVÊNCIA DE PACIENTES / EMPRESAS)
# =============================================================================

def run_survival_experiment():
    np.random.seed(42)
    N = 200
    
    # Gerando dados: X1 = Idade/Volatilidade, X2 = Tratamento/Setor
    X = np.random.randn(N, 2)
    # Risco Verdadeiro (Hazard Ratio): Idade aumenta risco, Tratamento reduz.
    beta_true = np.array([0.7, -1.2]) 
    
    # Gerando tempos de vida baseados no risco (distribuição de Weibull simplificada)
    hazard = np.exp(X @ beta_true)
    T = np.random.exponential(1.0 / hazard)
    # Censura: 20% dos dados não tiveram o evento registrado
    E = np.random.binomial(1, 0.8, N)
    
    # Ajustar Cox
    model = CoxPHScratch().fit(X, T, E)
    
    print(f"\nResultados:")
    print(f"Coeficientes Estimados: {model.beta}")
    print(f"Coeficientes Reais:     {beta_true}")
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    times, S0 = model.baseline_survival(X, T, E)
    
    # Survival Curves para dois perfis
    profile_low_risk = np.array([[-1, 1]])  # Jovem + Tratamento
    profile_high_risk = np.array([[1, -1]]) # Idoso - Tratamento
    
    S_low = S0 ** model.predict_hazard_ratio(profile_low_risk)
    S_high = S0 ** model.predict_hazard_ratio(profile_high_risk)
    
    plt.figure(figsize=(10, 6))
    plt.step(times, S_low, label="Baixo Risco (Ex: Paciente Jovem/Tratado)", color='#2ca02c', lw=2)
    plt.step(times, S_high, label="Alto Risco (Ex: Paciente Idoso/Sem Tratamento)", color='#d62728', lw=2)
    plt.step(times, S0, '--', label="Linha de Base", color='black', alpha=0.5)
    
    plt.title("Análise de Sobrevivência: Curvas de Sobrevivência Estimadas (Breslow)")
    plt.xlabel("Tempo (Anos / Meses)")
    plt.ylabel("Probabilidade de Sobrevivência S(t)")
    plt.legend()
    plt.savefig("figuras/survival_cox_results.png", dpi=300)
    plt.close()
    
    print("Figura salva em 'figuras/survival_cox_results.png'")

if __name__ == "__main__":
    run_survival_experiment()
