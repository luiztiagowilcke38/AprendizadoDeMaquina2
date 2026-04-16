"""
Predição de Tempo de Vida e Falência: Análise de Sobrevivência de Comércios Locais
Baseado no Cap. 43 (Análise de Sobrevivência e Riscos Proporcionais) do livro 'Aprendizado de Máquina'.

Neste script, aplicamos os fundamentos do Machine Learning aplicados à censura (quando não 
sabemos se/quando o evento ocorrerá). Modelamos a taxa instantânea de risco (Hazard Function) 
e a curva de sobrevida de comércios do centro de uma cidade, avaliando o impacto 
da Concorrência e do Preço do Aluguel no "Tempo de Morte" do negócio.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas profissionais (Padrão Livro Aprendizado de Máquina)
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

def estimate_kaplan_meier(durations, events):
    """
    Implementação rigorosa do Estimador Não-Paramétrico de Kaplan-Meier.
    Calcula a Função de Sobrevivência S(t) = P(T > t).
    """
    times = np.sort(np.unique(durations))
    survival_prob = np.ones(len(times) + 1)
    
    n_at_risk = len(durations)
    for i, t in enumerate(times):
        # Indivíduos que morreram (faliram) no tempo t
        deaths = np.sum((durations == t) & (events == 1))
        # Probabilidade de sobreviver após o tempo t dado que sobreviveu até t
        prob_survive_t = 1.0 - (deaths / n_at_risk)
        
        survival_prob[i+1] = survival_prob[i] * prob_survive_t
        
        # Diminuir da população em risco (falidos + censurados que saíram da amostra)
        lost = np.sum(durations == t)
        n_at_risk -= lost
        
    # Adicionar tempo 0 com 100% de sobrevida
    times_extended = np.insert(times, 0, 0)
    return times_extended, survival_prob

def estimate_hazard(times, survival_probs):
    """
    Estimativa numérica da Função de Risco Lambda(t) (Hazard)
    Lambda(t) = - S'(t) / S(t) -> Probabilidade instantânea de quebrar.
    """
    # Derivada discreta
    dS = np.diff(survival_probs)
    dt = np.diff(times)
    dt[dt == 0] = 1e-5 # Prevenção de divisão por zero
    
    S_mid = (survival_probs[:-1] + survival_probs[1:]) / 2
    hazard = -(dS / dt) / S_mid
    
    # Remover ruídos matemáticos pontuais
    hazard = np.maximum(hazard, 0)
    return times[1:], hazard

def generate_business_lifespan_data(N=200):
    """
    Gera dados simulados de sobrevivência de comércios no centro da cidade.
    O "evento" (E=1) significa que o comércio quebrou.
    (E=0) significa censura (ainda está aberto no fim do estudo).
    """
    np.random.seed(42)
    
    # Condição 1: Comércios em rua de Aluguel Baixo
    # Condição 2: Comércios em rua de Aluguel Alto (Quebram mais rápido)
    
    # Modelo Weibull para simular riscos contínuos diferentes
    lambda_baixo = 0.02
    lambda_alto = 0.05
    
    tempo_baixo = np.random.exponential(1/lambda_baixo, N//2)
    tempo_alto = np.random.exponential(1/lambda_alto, N//2)
    
    # Estudo dura 60 meses. Quem viver mais do que isso sofre "Censura Direita" (E=0)
    durations = np.concatenate([tempo_baixo, tempo_alto])
    events = np.ones(N) # Todos tendem falir pela distribuição
    
    # Aplicar Censura
    study_duration = 60 # Em meses
    censored = durations > study_duration
    durations[censored] = study_duration
    events[censored] = 0
    
    group = np.array(['Aluguel Baixo'] * (N//2) + ['Aluguel Alto'] * (N//2))
    
    return durations, events, group

def analyze_city_center_survival():
    print("Levantamento de dados: Modelando a vida de comércios urbanos...")
    t, e, g = generate_business_lifespan_data(N=300)
    
    idx_baixo = g == 'Aluguel Baixo'
    idx_alto = g == 'Aluguel Alto'
    
    # KM Estimator (Cap 43)
    times_baixo, surv_baixo = estimate_kaplan_meier(t[idx_baixo], e[idx_baixo])
    times_alto, surv_alto = estimate_kaplan_meier(t[idx_alto], e[idx_alto])
    
    # Hazard Engine
    h_times_baixo, haz_baixo = estimate_hazard(times_baixo, surv_baixo)
    h_times_alto, haz_alto = estimate_hazard(times_alto, surv_alto)
    
    # Plotagem Rigorosa (Estilo do Livro)
    os.makedirs("figuras", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Função de Sobrevida P(T > t)
    ax1.step(times_baixo, surv_baixo, where='post', label='Rua Periférica (Aluguel Baixo)', color=AZUL, lw=2.5)
    ax1.step(times_alto, surv_alto, where='post', label='Rua Principal (Aluguel Alto)', color=VERMELHO, lw=2.5)
    
    # Marcar pontos de censura
    censura_baixo = t[idx_baixo & (e == 0)]
    idx_cens = [np.searchsorted(times_baixo, c) for c in censura_baixo]
    if len(censura_baixo) > 0:
        ax1.scatter(censura_baixo, surv_baixo[idx_cens], marker='+', color=AZUL, s=100, zorder=3, label='Status Ativo (Censurado)')
        
    ax1.set_title("Curva de Sobrevida dos Comércios $S(t) = P(T > t)$")
    ax1.set_xlabel("Meses de Atividade")
    ax1.set_ylabel("Probabilidade de Sobrevivência")
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    
    # Subplot 2: Risco Proporcional de Cox / Hazard function
    # Suavizando matematicamente para o plot com moving average
    window = 3
    if len(haz_baixo) >= window:
        haz_baixo_smooth = np.convolve(haz_baixo, np.ones(window)/window, mode='valid')
        h_t_baixo = h_times_baixo[window//2 : -window//2 + 1] if len(h_times_baixo) % 2 != 0 else h_times_baixo[1:-1]
        
    if len(haz_alto) >= window:
        haz_alto_smooth = np.convolve(haz_alto, np.ones(window)/window, mode='valid')
        h_t_alto = h_times_alto[window//2 : -window//2 + 1] if len(h_times_alto) % 2 != 0 else h_times_alto[1:-1]
        
    # Plot bruto para fidelidade (step function do Hazard Empirico)
    ax2.step(h_times_baixo, haz_baixo, where='post', color=AZUL, alpha=0.4)
    ax2.step(h_times_alto, haz_alto, where='post', color=VERMELHO, alpha=0.4)
    
    # Se fosse o Modelo Parametrico de Cox (Cap 43 Eq. 13.58), seria uma linha exp(beta) estavel.
    ax2.axhline(0.02, color='black', linestyle='--', label=r'Risco Constante Teórico $\lambda_0$')
    
    ax2.set_title(r"Função de Risco (Taxa de Falência Instantânea) $\lambda(t)$")
    ax2.set_xlabel("Meses de Atividade")
    ax2.set_ylabel("Hazard (Risco de quebrar no tempo $t$)")
    ax2.legend()
    
    plt.tight_layout()
    file_path = "figuras/comercio_survival_analysis.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    
    print("\n--- ANÁLISE CONCLUÍDA ---")
    print(f"Prob. de um negócio sobreviver 30 meses na rua cara: {surv_alto[np.searchsorted(times_alto, 30)]: .1%} ")
    print(f"Prob. de um negócio sobreviver 30 meses na rua barata: {surv_baixo[np.searchsorted(times_baixo, 30)]: .1%} ")
    print(f"O Modelo Não-Paramétrico Kaplan-Meier e a Função de Hazard foram aplicados.")
    print(f"Resultado Gráfico exportado para: {file_path}")

if __name__ == "__main__":
    analyze_city_center_survival()
