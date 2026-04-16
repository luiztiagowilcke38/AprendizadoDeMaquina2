"""
Climatologia e Aprendizado de Máquina: Detecção de Tendências e Anomalias Climáticas
Baseado no Cap. 80 (Regressão Polinomial Local e Viés) do livro 'Aprendizado de Máquina'.

Em Climatologia, séries de temperatura possuem sazonalidade, tendências de longo prazo e
ruído estocástico combinados. Uma abordagem clássica — mas matematicamente ingênua — é
aplicar uma regressão global (polinomial ou linear) sobre toda a série, gerando o problema
do "trimming the hills and filling the valleys": picos de calor e vales frios são suavizados
erroneamente por um modelo global que ignora a curvatura local.

O Cap. 80 resolve isso via Regressão Polinomial Local (LOWESS): em cada instante de tempo,
ajustamos um polinômio de baixo grau APENAS para os dados vizinhos, ponderados pelo seu
kernel de distância. O resultado é um estimador de tendência preciso sem viés de borda.
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
VERMELHO = '#d62728'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'
CINZA = '#aaaaaa'


# =========================================================================
# 1. KERNEL DE SUAVIZAÇÃO (Tricúbico — padrão LOWESS)
# Conforme Cap. 80: K_lambda(x0, xi) define os pesos dos vizinhos locais.
# =========================================================================

def tricubic_kernel(distances, h):
    """
    Kernel Tricúbico: K(u) = (1 - |u|^3)^3, se |u| < 1
    Concentra o peso nos vizinhos mais próximos de x0.
    """
    u = distances / h
    u = np.clip(u, 0, 1)
    return (1.0 - u**3)**3


# =========================================================================
# 2. REGRESSÃO POLINOMIAL LOCAL (Cap. 80 Eq. 6.11)
#    Para cada ponto x0, resolve mínimos quadrados ponderados pelo kernel.
# =========================================================================

def local_poly_fit(x0, X, y, h, degree=1):
    """
    Estima f(x0) via polinômio local de grau 'degree':
    min sum_i K(x0, xi) * [yi - sum_j beta_j * xi^j]^2
    """
    distances = np.abs(X - x0)
    weights = tricubic_kernel(distances, h)

    # Montar a matriz de design polinomial local
    # [1, xi, xi^2, ..., xi^degree] centrada em x0
    X_centered = X - x0
    design = np.column_stack([X_centered**j for j in range(degree + 1)])

    # Mínimos quadrados ponderados (WLSR): beta = (D^T W D)^{-1} D^T W y
    W = np.diag(weights)
    DtW = design.T.dot(W)
    try:
        beta = np.linalg.solve(DtW.dot(design) + 1e-8 * np.eye(degree + 1), DtW.dot(y))
    except np.linalg.LinAlgError:
        beta = np.zeros(degree + 1)

    # A predição em x0 é apenas o intercepto (termo zero, pois centramos em x0)
    return beta[0]


def lowess_climatology(X, y, h_frac=0.15, degree=1, n_eval=None):
    """
    Aplica Regressão Polinomial Local sobre toda a série temporal.
    h_frac: fração da janela de suavização (controla o tradeoff viés-variância, Cap 80, Eq. 6.11)
    """
    h = h_frac * (X.max() - X.min())
    X_eval = X if n_eval is None else np.linspace(X.min(), X.max(), n_eval)
    trend = np.array([local_poly_fit(x0, X, y, h, degree) for x0 in X_eval])
    return X_eval, trend


# =========================================================================
# 3. GERAÇÃO DE SÉRIE CLIMÁTICA SIMULADA
# =========================================================================

def generate_climate_series(n_years=40):
    """
    Gera uma série mensal de temperatura simulada com:
    - Sazonalidade Anual (padrão cosseno)
    - Tendência de Aquecimento Global (linear + nonlinear)
    - Anomalias Climáticas (El Niño simulado)
    - Ruído Estocástico
    """
    np.random.seed(42)
    n_months = n_years * 12
    t = np.linspace(0, n_years, n_months)  # Em anos

    # Tendência de aquecimento não-linear
    warming_trend = 0.03 * t + 0.001 * t**2

    # Sazonalidade anual
    seasonal = 12 * np.cos(2 * np.pi * t - np.pi / 3)

    # El Niño: picos de calor anômalos a cada ~7 anos
    el_nino = np.zeros(n_months)
    for peak in range(0, n_years, 7):
        center = int(peak * 12 + 18)
        if center < n_months:
            el_nino += 2.5 * np.exp(-0.5 * ((t - peak - 1.5) / 0.4)**2)

    # Ruído
    noise = 1.8 * np.random.randn(n_months)

    temperature = warming_trend + seasonal + el_nino + noise + 22  # Baseline 22°C
    return t, temperature, warming_trend + 22


def analyze_climate_data():
    print("Gerando série climática simulada (40 anos, resolução mensal)...")
    t, temp, true_trend = generate_climate_series(n_years=40)

    print("Ajustando Regressão Polinomial Local (LOWESS, Cap. 80)...")

    # 1. Estimativa de tendência suavizada com janela estreita (h=10% — capta mais detalhes)
    t_eval, trend_narrow = lowess_climatology(t, temp, h_frac=0.10, degree=1)

    # 2. Estimativa de longo prazo com janela larga (h=30% — tendência climática global)
    _, trend_wide = lowess_climatology(t, temp, h_frac=0.30, degree=1)

    # 3. Anomalias = temperatura observada - tendência suavizada
    anomalias = temp - trend_narrow

    # Identificar anomalias extremas (>2σ)
    sigma = np.std(anomalias)
    extremas_pos = t[anomalias > 2*sigma]
    extremas_neg = t[anomalias < -2*sigma]

    # 4. Visualização
    os.makedirs("figuras", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Série + Tendências LOWESS
    ax1.plot(t, temp, color=CINZA, lw=0.8, alpha=0.7, label='Temperatura Observada (°C)')
    ax1.plot(t_eval, trend_narrow, color=AZUL, lw=2.0, label=r'LOWESS Local ($h_{frac}=0.10$) — Tendência Anual')
    ax1.plot(t_eval, trend_wide, color=VERMELHO, lw=2.5, linestyle='--', label=r'LOWESS Global ($h_{frac}=0.30$) — Aquecimento de Longo Prazo')
    ax1.plot(t, true_trend, color=VERDE, lw=1.2, linestyle=':', label='Tendência Real Oculta (Referência)')
    ax1.set_title("Extração de Tendência Climática via Regressão Polinomial Local (Cap. 80)\nSérie Mensal de Temperatura — 40 Anos")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.legend(fontsize=8)

    # Subplot 2: Anomalias Climáticas + Detecção de Extremos
    ax2.bar(t, anomalias, color=[VERMELHO if a > 0 else AZUL for a in anomalias], alpha=0.6, width=0.08)
    ax2.axhline(2*sigma, color=LARANJA, lw=1.5, linestyle='--', label=f'Limiar +2σ ({2*sigma:.1f}°C)')
    ax2.axhline(-2*sigma, color=VERDE, lw=1.5, linestyle='--', label=f'Limiar -2σ ({-2*sigma:.1f}°C)')

    for ex in extremas_pos:
        ax2.axvline(ex, color=LARANJA, alpha=0.3, lw=0.8)
    for ex in extremas_neg:
        ax2.axvline(ex, color=VERDE, alpha=0.3, lw=0.8)

    ax2.set_title("Anomalias Climáticas Residuais (Observado - Tendência LOWESS)\nVermelhos = Calor Anômalo | Azuis = Frio Anômalo")
    ax2.set_xlabel("Ano")
    ax2.set_ylabel("Anomalia (°C)")
    ax2.legend()

    plt.tight_layout()
    file_path = "figuras/climatologia_lowess.png"
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"\nTendência de aquecimento estimada: {(trend_wide[-1] - trend_wide[0]):.2f}°C em 40 anos")
    print(f"Anomalias extremas quentes detectadas: {len(extremas_pos)}")
    print(f"Anomalias extremas frias detectadas:  {len(extremas_neg)}")
    print(f"\nGráfico exportado para: {file_path}")


if __name__ == "__main__":
    analyze_climate_data()
