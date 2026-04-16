"""
Previsão do Preço do Barril de Petróleo: Gradient Boosting com Perda de Huber
Baseado no Cap. 52 (GBM: Perdas Robustas — L1 e Huber) do livro 'Aprendizado de Máquina'.

Séries de preço de petróleo são notoriamente não-Gaussianas: possuem quedas abruptas
(crashes geopolíticos, crises financeiras) e picos extremos que violam a premissa da
perda quadrática (L2). O Cap. 52 demonstra que a Perda de Huber resolve exatamente isso,
combinando suavidade de L2 para erros pequenos com robustez de L1 para outliers extremos,
via parâmetro delta adaptativo.

Além disso, a Regressão Quantílica (também do Cap. 52) permite construir intervalos de
predição — prevendo não só o preço esperado, mas sua distribuição de risco.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
VERDE = '#2ca02c'
VERMELHO = '#d62728'
LARANJA = '#ff7f0e'
CINZA = '#aaaaaa'


# =========================================================================
# 1. PSEUDO-RESÍDUO DE HUBER (Cap. 52, Eq. 10.33)
#    Calcula o gradiente negativo da perda de Huber em cada iteração do GBM
# =========================================================================

def huber_gradient(y, f, delta):
    """
    Pseudo-resíduo Huber:
    - Se |y - f| <= delta: gradiente = (y - f)  [como L2]
    - Se |y - f| >  delta: gradiente = delta * sign(y - f)  [como L1]
    """
    residuals = y - f
    abs_res = np.abs(residuals)
    grad = np.where(abs_res <= delta, residuals, delta * np.sign(residuals))
    return grad


def huber_loss(y, f, delta):
    residuals = np.abs(y - f)
    loss = np.where(residuals <= delta,
                    0.5 * residuals**2,
                    delta * (residuals - 0.5 * delta))
    return np.mean(loss)


# =========================================================================
# 2. STUMP DE REGRESSÃO (Árvore de Profundidade 1, "aprendiz fraco")
# =========================================================================

class RegressionStump:
    """Aprendiz fraco: árvore com apenas uma divisão binária."""
    def __init__(self):
        self.split_val = None
        self.left_val = None
        self.right_val = None
        self.feature = 0

    def fit(self, X, r):
        best_loss = np.inf
        n = len(r)
        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            for thr in thresholds:
                left = r[X[:, j] <= thr]
                right = r[X[:, j] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                loss = np.var(left) * len(left) + np.var(right) * len(right)
                if loss < best_loss:
                    best_loss = loss
                    self.split_val = thr
                    self.feature = j
                    # Saída: mediana (robusta a outliers, consistente com Huber)
                    self.left_val = np.median(left)
                    self.right_val = np.median(right)

    def predict(self, X):
        preds = np.where(X[:, self.feature] <= self.split_val,
                         self.left_val, self.right_val)
        return preds


# =========================================================================
# 3. GRADIENT BOOSTING COM PERDA DE HUBER (Cap. 52)
# =========================================================================

class HuberGBM:
    """
    Gradient Boosting com Perda de Huber implementado do zero.
    Em cada iteração m:
      1. Calcula pseudo-resíduos via gradiente de Huber
      2. Ajusta stump aos pseudo-resíduos
      3. Atualiza o ensemble via passo de shrinkage (learning rate)
    """
    def __init__(self, n_estimators=150, lr=0.1, delta_quantile=0.9):
        self.n_estimators = n_estimators
        self.lr = lr
        self.delta_q = delta_quantile
        self.stumps = []
        self.f0 = None
        self.losses = []
        self.deltas = []

    def fit(self, X, y):
        # Inicialização: mediana (ótimo para L1)
        self.f0 = np.median(y)
        F = np.full(len(y), self.f0)

        for m in range(self.n_estimators):
            # Delta adaptativo: quantil dos resíduos absolutos desta iteração
            delta = np.quantile(np.abs(y - F), self.delta_q)
            self.deltas.append(delta)

            # Pseudo-resíduos Huber
            r = huber_gradient(y, F, delta)

            # Ajustar stump ao pseudo-resíduo
            stump = RegressionStump()
            stump.fit(X, r)

            # Atualizar modelo
            F += self.lr * stump.predict(X)
            self.stumps.append(stump)
            self.losses.append(huber_loss(y, F, delta))

    def predict(self, X):
        F = np.full(X.shape[0], self.f0)
        for stump in self.stumps:
            F += self.lr * stump.predict(X)
        return F


# =========================================================================
# 4. REGRESSÃO QUANTÍLICA (Cap. 52, Eq. 10.36 — Pinball Loss)
# =========================================================================

def pinball_gradient(y, f, q):
    """Pseudo-resíduo da Perda Pinball para quantil q."""
    return np.where(y >= f, q * np.ones_like(y), -(1 - q) * np.ones_like(y))


class QuantileGBM:
    """GBM com Perda Pinball para previsão de intervalos de risco."""
    def __init__(self, quantile=0.90, n_estimators=100, lr=0.1):
        self.q = quantile
        self.n_estimators = n_estimators
        self.lr = lr
        self.stumps = []
        self.f0 = None

    def fit(self, X, y):
        self.f0 = np.quantile(y, self.q)
        F = np.full(len(y), self.f0)
        for _ in range(self.n_estimators):
            r = pinball_gradient(y, F, self.q)
            stump = RegressionStump()
            stump.fit(X, r)
            F += self.lr * stump.predict(X)
            self.stumps.append(stump)

    def predict(self, X):
        F = np.full(X.shape[0], self.f0)
        for stump in self.stumps:
            F += self.lr * stump.predict(X)
        return F


# =========================================================================
# 5. DADOS E EXPERIMENTO
# =========================================================================

def generate_oil_prices(n=600):
    """
    Simula série de preço de petróleo (USD/barril) com:
    - Tendência lenta e não-linear
    - Ciclos de oferta/demanda
    - Crashes abruptos (outliers positivos no erro — capturados por Huber)
    """
    np.random.seed(42)
    t = np.linspace(0, 20, n)

    trend = 60 + 0.5*t + 4*np.sin(0.8*t) + 2*np.sin(0.3*t)
    noise = 5 * np.random.randn(n)

    # Crashes abruptos (crise geopolítica / pandemia)
    crashes = np.zeros(n)
    for crash_t in [4.0, 8.5, 13.0, 16.5]:
        idx = np.argmin(np.abs(t - crash_t))
        crashes[max(0, idx-5):idx+15] -= np.random.uniform(20, 45)

    price = trend + noise + crashes
    price = np.clip(price, 5, 150)
    return t, price

def build_features(t, price, lookback=12):
    """Constrói janela de features: lags do preço."""
    X, y = [], []
    for i in range(lookback, len(price)):
        X.append(price[i-lookback:i])
        y.append(price[i])
    return np.array(X), np.array(y), t[lookback:]

def analyze_oil_prices():
    print("Gerando série de preço de petróleo simulada...")
    t_full, price = generate_oil_prices(n=600)
    X, y, t = build_features(t_full, price, lookback=12)

    # Split treino/teste
    split = int(0.75 * len(y))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    t_tr, t_te = t[:split], t[split:]

    # Treinar modelos
    print("Treinando GBM com Perda de Huber (Cap. 52)...")
    gbm = HuberGBM(n_estimators=200, lr=0.08, delta_quantile=0.85)
    gbm.fit(X_tr, y_tr)

    print("Treinando Regressão Quantílica (Intervalo de Risco, Cap. 52)...")
    gbm_q10 = QuantileGBM(quantile=0.10, n_estimators=150, lr=0.08)
    gbm_q90 = QuantileGBM(quantile=0.90, n_estimators=150, lr=0.08)
    gbm_q10.fit(X_tr, y_tr)
    gbm_q90.fit(X_tr, y_tr)

    pred_te = gbm.predict(X_te)
    q10_te = gbm_q10.predict(X_te)
    q90_te = gbm_q90.predict(X_te)

    # Métricas
    mae = np.mean(np.abs(pred_te - y_te))
    rmse = np.sqrt(np.mean((pred_te - y_te)**2))
    coverage = np.mean((y_te >= q10_te) & (y_te <= q90_te)) * 100

    # Visualização
    os.makedirs("figuras", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(t_tr, y_tr, color=CINZA, lw=1, alpha=0.7, label='Preço Treino (USD/barril)')
    ax1.plot(t_te, y_te, color='black', lw=1.5, label='Preço Real (Teste)')
    ax1.plot(t_te, pred_te, color=AZUL, lw=2.5, label='GBM Huber — Previsão Mediana')
    ax1.fill_between(t_te, q10_te, q90_te, alpha=0.25, color=LARANJA,
                     label=f'Intervalo de Risco 80% (Regressão Quantílica)')
    ax1.axvline(t_te[0], color='red', linestyle='--', lw=1.5, label='Início do Período de Teste')
    ax1.set_title("Previsão do Preço do Barril de Petróleo — GBM com Perda de Huber\n(Cap. 52: Perdas Robustas e Regressão Quantílica)")
    ax1.set_ylabel("Preço (USD/barril)")
    ax1.legend(fontsize=8)

    ax2.semilogy(gbm.losses, color=VERMELHO, lw=2)
    ax2.set_title("Convergência da Perda de Huber ao Longo das Iterações do Boosting")
    ax2.set_xlabel("Iteração (Árvore Adicionada)")
    ax2.set_ylabel("Huber Loss (escala log)")

    plt.tight_layout()
    path = "figuras/petroleo_huber_gbm.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"\nMAE no Teste:  {mae:.2f} USD/barril")
    print(f"RMSE no Teste: {rmse:.2f} USD/barril")
    print(f"Cobertura do Intervalo 80%: {coverage:.1f}% (esperado: ~80%)")
    print(f"\nGráfico exportado para: {path}")

if __name__ == "__main__":
    analyze_oil_prices()
