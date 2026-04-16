"""
Aprendizado por Reforço em Investimentos: Agente de Trading (Q-Learning)
Baseado no Cap. 11 do livro 'Aprendizado de Máquina'.

Este script implementa um ambiente de trading simplificado e um agente de Q-Learning
que aprende a maximizar o retorno financeiro através da interação e recompensa,
aplicando os conceitos de Tomada de Decisão sob Incerteza do Capítulo 11.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas profissionais (Seguindo o padrão do livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'
VERMELHO = '#d62728'

# 1. AMBIENTE DE TRADING (Processo de Decisão de Markov)
class TradingEnv:
    """
    Simula um mercado financeiro para um agente de RL.
    Estado: Momentum do preço (Tendência).
    Ações: 0 (Fora do Mercado), 1 (Comprado/Long).
    """
    def __init__(self, prices, window_size=5):
        self.prices = prices
        self.n_steps = len(prices)
        self.window_size = window_size
        self.current_step = 0
        self.states = self._discretize_states()
        
    def _discretize_states(self):
        # O estado é definido pelo momentum: média dos retornos recentes
        returns = np.diff(self.prices) / self.prices[:-1]
        momentum = np.zeros(len(self.prices))
        for i in range(self.window_size, len(self.prices)):
            momentum[i] = np.mean(returns[i-self.window_size:i])
        
        # Discretizar o momentum em 10 níveis (Estados do MDP)
        # Permite que uma Q-Table finita aprenda a relação Tendência -> Ação
        bins = np.linspace(momentum.min(), momentum.max(), 10)
        return np.digitize(momentum, bins) - 1

    def reset(self):
        self.current_step = self.window_size
        return self.states[self.current_step]

    def step(self, action):
        price_now = self.prices[self.current_step]
        price_next = self.prices[self.current_step + 1]
        daily_return = (price_next - price_now) / price_now
        
        # Recompensa (Reward): Retorno aritmético se estiver "comprado"
        # Conforme Cap. 11, a recompensa guia a otimização da política
        reward = daily_return if action == 1 else 0
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 2
        next_state = self.states[self.current_step]
        
        return next_state, reward, done

# 2. AGENTE Q-LEARNING (Equação de Bellman)
class QLearningAgent:
    """Implementação do algoritmo Q-Learning (Cap. 11.3)."""
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95, epsilon=1.0):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr           # Taxa de aprendizado (Alpha)
        self.gamma = gamma     # Fator de desconto para recompensas futuras
        self.epsilon = epsilon # Exploração inicial
        self.epsilon_decay = 0.998
        self.min_epsilon = 0.05

    def choose_action(self, state):
        # Estratégia Epsilon-Greedy (Seção 5 do Cap. 11)
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        # Atualização da Q-Table via Erro de Diferença Temporal (TD)
        predict = self.q_table[state, action]
        # Target baseado na Equação de Otimalidade de Bellman
        target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.lr * (target - predict)
        
        # Decaimento de exploração para convergência
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# 3. EXECUÇÃO DO EXPERIMENTO
def run_financial_rl_simulation():
    np.random.seed(42)
    
    # 3.1 Gerar Preços (Movimento Browniano com Drift e Volatilidade)
    T = 600
    mu = 0.0006 # Drift positivo (0.06% ao dia)
    sigma = 0.012 # Volatilidade (1.2% ao dia)
    prices = [100]
    for _ in range(T-1):
        prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.randn()))
    prices = np.array(prices)

    env = TradingEnv(prices)
    agent = QLearningAgent(n_states=10, n_actions=2) # 10 níveis de momentum, 2 ações
    
    # 3.2 Fase de Treinamento (Iterações de Bellman)
    print("Treinando Agente de RL para o Mercado...")
    n_episodes = 500
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            
    # 3.3 Fase de Teste e Avaliação (Backtesting)
    agent.epsilon = 0 
    state = env.reset()
    done = False
    portfolio_rl = [100.0]
    portfolio_bh = [100.0] # Benchmark: Buy & Hold
    positions = []
    
    while not done:
        action = agent.choose_action(state)
        positions.append(action)
        next_state, reward, done = env.step(action)
        
        portfolio_rl.append(portfolio_rl[-1] * (1 + reward))
        
        # Benchmark Real
        price_now = prices[env.current_step-1]
        price_next = prices[env.current_step]
        bh_ret = (price_next - price_now) / price_now
        portfolio_bh.append(portfolio_bh[-1] * (1 + bh_ret))
        
        state = next_state

    # 3.4 Visualização Profissional
    os.makedirs("figuras", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Ativo e Ações do Agente
    time_axis = range(len(positions))
    ax1.plot(prices[env.window_size:-1], color='gray', alpha=0.4, label='Preço do Ativo')
    
    # Destacar onde o agente está posicionado
    buy_signals = [i for i, action in enumerate(positions) if action == 1]
    ax1.scatter(buy_signals, prices[env.window_size:-1][buy_signals], 
                color=VERDE, marker='o', s=5, label='Agente Comprado', alpha=0.6)
    ax1.set_title("Dinâmica de Operação do Agente de Q-Learning (Cap. 11)")
    ax1.set_ylabel("Preço")
    ax1.legend()
    
    # Subplot 2: Comparativo de Rentabilidade
    ax2.plot(portfolio_rl, color=AZUL, lw=2.5, label='Estratégia Aprendizado por Reforço')
    ax2.plot(portfolio_bh, color=VERMELHO, lw=1.5, linestyle='--', label='Estratégia Passiva (Buy & Hold)')
    ax2.set_title("Curva de Patrimônio (Equity Curve): RL vs Benchmark")
    ax2.set_xlabel("Passos Temporais (Dias)")
    ax2.set_ylabel("Valor da Carteira ($)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("figuras/trading_rl_results.png", dpi=300)
    plt.close()
    
    # Resultados Numéricos
    ret_rl = (portfolio_rl[-1] / portfolio_rl[0] - 1) * 100
    ret_bh = (portfolio_bh[-1] / portfolio_bh[0] - 1) * 100
    print("\n" + "="*40)
    print(f"ANÁLISE DE PERFORMANCE (RL INVEST):")
    print(f"Retorno Final Agente RL: {ret_rl:.2f}%")
    print(f"Retorno Final Buy & Hold: {ret_bh:.2f}%")
    print(f"Alpha do Agente: {ret_rl - ret_bh:.2f}%")
    print("="*40)
    print("Visualização gerada em 'figuras/trading_rl_results.png'")

if __name__ == "__main__":
    run_financial_rl_simulation()
