"""
Previsão de Riscos de Enchente via Random Forest
Baseado no Cap. 7 do livro 'Aprendizado de Máquina' (Árvores e Florestas).

Este script utiliza um conjunto de árvores de decisão para classificar o nível
de alerta ambiental (Baixo, Moderado, Emergência) baseado em sensores 
pluviométricos e fluviométricos, demonstrando a robustez dos modelos de ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Configurações estéticas profissionais (Padrão do Livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERMELHO = '#d62728'

# 1. SIMULAÇÃO DE DADOS HIDROMETEOROLÓGICOS
def generate_flood_data(n_samples=1200):
    """
    Simula sensores meteorológicos para previsão de desastres.
    """
    np.random.seed(42)
    # Features: Chuva Acumulada (mm), Umidade do Solo (%), Nível do Rio (m), Topografia (0-1)
    chuva = np.random.gamma(2, 25, n_samples) # Chuvas torrenciais (cauda longa)
    umidade = np.random.uniform(15, 100, n_samples)
    rio = np.random.normal(1.5, 0.4, n_samples) + (0.04 * chuva) # Correlação física
    topografia = np.random.uniform(0, 1, n_samples) # 0: Planície, 1: Encosta
    
    X = np.column_stack([chuva, umidade, rio, topografia])
    
    # Lógica de Risco (Ground Truth não linear)
    # O risco aumenta drasticamente com a combinação de chuva forte e solo saturado
    score = (0.35 * chuva/100) + (0.3 * rio/5) + (0.25 * umidade/100) + (0.1 * (1 - topografia))
    
    y = np.zeros(n_samples)
    y[score > 0.42] = 1 # Risco Moderado (Alerta)
    y[score > 0.62] = 2 # Risco Alto (Emergência/Transbordamento)
    
    return X, y, ["Precipitação (mm)", "Umidade Solo (%)", "Nível do Rio (m)", "Topografia"]

# 2. TREINAMENTO DO ENSEMBLE (Cap. 7.3: Random Forests)
def train_flood_risk_model():
    print("Coletando dados simulados de sensores ambientais...")
    X, y, features = generate_flood_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Floresta Aleatória: B=100 árvores para redução da variância via Bagging (Eq. 7.15)
    # max_depth=7 para evitar overfitting e manter generalização
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    
    accuracy = rf.score(X_test, y_test)
    print(f"Modelo treinado com sucesso. Acurácia de teste: {accuracy*100:.2f}%")
    
    # --- VISUALIZAÇÃO DOS RESULTADOS ---
    os.makedirs("figuras", exist_ok=True)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    
    # A. Fronteira de Decisão (Chuva vs Nível do Rio)
    ax1 = fig.add_subplot(gs[0, 0])
    xx, yy = np.meshgrid(np.linspace(0, 160, 100), np.linspace(1, 6, 100))
    # Projetar as outras variáveis para as médias para visualização 2D
    X_grid = np.column_stack([xx.ravel(), 
                             np.full(10000, np.mean(X[:,1])), 
                             yy.ravel(), 
                             np.full(10000, np.mean(X[:,3]))])
    Z = rf.predict(X_grid).reshape(xx.shape)
    
    # Background de risco colorido
    cmap_risk = plt.cm.get_cmap('RdYlGn_r')
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_risk)
    scatter = ax1.scatter(X_test[:,0], X_test[:,2], c=y_test, cmap=cmap_risk, edgecolor='k', s=25, alpha=0.7)
    ax1.set_title("Mapeamento de Risco Ambiental: Chuva vs Rio\n(Fronteiras aprendidas pela Random Forest)")
    ax1.set_xlabel(features[0])
    ax1.set_ylabel(features[2])
    
    # B. Importância de Características (Cap. 7.80)
    ax2 = fig.add_subplot(gs[0, 1])
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)
    ax2.barh(range(len(features)), importances[sorted_idx], color=AZUL, alpha=0.8)
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels([features[i] for i in sorted_idx])
    ax2.set_title("Determinantes do Risco (Feature Importance)\nIdentificação dos gatilhos críticos de inundação")
    ax2.set_xlabel("Importância (Gini Decrease)")
    
    # C. Matriz de Confusão (Análise de Desempenho)
    ax3 = fig.add_subplot(gs[1, :])
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=["Normal", "Alerta", "Emergência"])
    disp.plot(ax=ax3, cmap='Blues', colorbar=False)
    ax3.set_title("Matriz de Confusão: Avaliação dos Alertas de Defesa Civil")
    
    plt.tight_layout()
    plt.savefig("figuras/flood_risk_analysis.png", dpi=300)
    plt.close()
    
    print("\n" + "="*50)
    print("RESUMO DA PREVISÃO DE ENCHENTES:")
    print(f"Fator mais decisivo: {features[np.argmax(importances)]}")
    print(f"Contribuição da Precipitação: {importances[0]*100:.1f}%")
    print("="*50)
    print("Figura de análise gerada: 'figuras/flood_risk_analysis.png'")

if __name__ == "__main__":
    train_flood_risk_model()
