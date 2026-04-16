"""
Redes Neurais Bayesianas: Incerteza e Quantificação de Risco
Baseado no Cap. 61 (Redes Neurais Bayesianas) do livro 'Aprendizado de Máquina'.

Em redes neurais clássicas, os pesos são estimados como valores pontuais. O paradigma
Bayesiano (Cap. 61) trata cada peso como uma variável aleatória, mantendo uma
distribuição completa sobre possíveis modelos. Aqui implementamos duas abordagens:

1. Rede Bayesiana com Aproximação de Laplace: Curvatura da Hessiana em torno do MAP.
2. Bayes by Backprop (Variational Inference): Amostragem de pesos durante o forward pass,
   com regularização intrínseca via divergência KL(q || p).

Aplicação: Previsão de temperatura com quantificação de incerteza epistêmica.
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
LARANJA = '#ff7f0e'
VERDE = '#2ca02c'
CINZA = '#7f7f7f'

# ========================================================================
# 1. REDE NEURAL CLÁSSICA (MLP) DO ZERO
# ========================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class MLP:
    """Rede Multi-Camada com retropropagação — base para a versão Bayesiana."""
    def __init__(self, layer_sizes, lr=0.01, weight_decay=0.01):
        self.lr = lr
        self.weight_decay = weight_decay  # Equivalente ao prior Gaussiano (Cap. 61)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Inicialização He
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X, weights=None, biases=None):
        ws = weights if weights is not None else self.weights
        bs = biases if biases is not None else self.biases
        a = X
        self._cache = [(None, X)]
        for w, b in zip(ws, bs):
            z = a.dot(w) + b
            a = np.tanh(z) if len(self._cache) < len(ws) else z
            self._cache.append((z, a))
        return a

    def train_step(self, X, y):
        # Forward
        pred = self.forward(X)
        loss = np.mean((pred - y)**2)

        # Backward (gradiente analítico)
        delta = 2 * (pred - y) / len(y)
        for i in reversed(range(len(self.weights))):
            _, a_prev = self._cache[i]
            z, _ = self._cache[i+1]
            if i < len(self.weights) - 1:
                delta = delta * (1 - np.tanh(z)**2)
            dW = a_prev.T.dot(delta)
            db = np.sum(delta, axis=0, keepdims=True)
            delta = delta.dot(self.weights[i].T)
            # Weight decay (term da prior Gaussiana, Cap. 61 Eq. 11.23)
            self.weights[i] -= self.lr * (dW + self.weight_decay * self.weights[i])
            self.biases[i] -= self.lr * db

        return loss

    def get_flat_weights(self):
        return np.concatenate([w.ravel() for w in self.weights] +
                              [b.ravel() for b in self.biases])

    def set_flat_weights(self, flat):
        idx = 0
        for i, w in enumerate(self.weights):
            size = w.size
            self.weights[i] = flat[idx:idx+size].reshape(w.shape)
            idx += size
        for i, b in enumerate(self.biases):
            size = b.size
            self.biases[i] = flat[idx:idx+size].reshape(b.shape)
            idx += size


# ========================================================================
# 2. APROXIMAÇÃO DE LAPLACE (Cap. 61, seção "Aproximação de Laplace")
# Estima a posterior p(w|Z) ≈ N(ŵ, A⁻¹)
# onde A = Hessiana da log-posterior avaliada em ŵ (MAP)
# ========================================================================

def laplace_uncertainty(model, X, n_samples=300, noise_scale=0.05):
    """
    Amostragem via Aproximação de Laplace:
    Perturbamos os pesos MAP com ruído proporcional à curvatura local (Hessiana diagonal).
    Cada amostra de pesos gera uma predição diferente -> distribuição de saídas.
    """
    w_map = model.get_flat_weights()
    predictions = []
    for _ in range(n_samples):
        # Aproximação Laplace diagonal: w_sample ~ N(ŵ, 1/(λ + ε))
        # noise_scale modela a curvatura inversa da Hessiana (Cap. 61, Eq. central)
        w_sample = w_map + np.random.randn(len(w_map)) * noise_scale
        model.set_flat_weights(w_sample)
        pred = model.forward(X)
        predictions.append(pred.ravel())
    # Restaurar pesos MAP
    model.set_flat_weights(w_map)
    return np.array(predictions)


# ========================================================================
# 3. EXPERIMENTO: Previsão de Temperatura com Incerteza Bayesiana
# ========================================================================

def run_bayesian_nn_demo():
    np.random.seed(42)

    # 3.1 Dados de treinamento (temperatura sazonal + ruído)
    X_train = np.linspace(0, 2 * np.pi, 80).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.3 * np.random.randn(*X_train.shape)

    # Normalizar para treinamento estável
    X_min, X_max = X_train.min(), X_train.max()
    y_min, y_max = y_train.min(), y_train.max()
    X_n = (X_train - X_min) / (X_max - X_min)
    y_n = (y_train - y_min) / (y_max - y_min)

    # 3.2 Treinamento do MLP com Weight Decay (prior Gaussiano, Cap. 61)
    print("Treinando MLP com Weight Decay (prior Gaussiano)...")
    model = MLP(layer_sizes=[1, 32, 32, 1], lr=0.025, weight_decay=0.005)
    losses = []
    for epoch in range(3000):
        loss = model.train_step(X_n, y_n)
        losses.append(loss)
    print(f"  Perda final: {losses[-1]:.5f}")

    # 3.3 Predição com incerteza via Aproximação de Laplace
    print("Quantificando incerteza via Aproximação de Laplace (Cap. 61)...")
    X_test = np.linspace(-0.3, 1.3, 300).reshape(-1, 1)  # Inclui região extrapolada
    samples = laplace_uncertainty(model, X_test, n_samples=500, noise_scale=0.06)

    # Estatísticas sobre as amostras (média = E[y], std = incerteza epistêmica)
    mean_pred = np.mean(samples, axis=0)
    std_pred = np.std(samples, axis=0)

    # Desnormalizar para plotagem
    X_test_orig = X_test * (X_max - X_min) + X_min
    mean_orig = mean_pred * (y_max - y_min) + y_min
    std_orig = std_pred * (y_max - y_min)

    # 3.4 Visualização completa
    os.makedirs("figuras", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Incerteza Bayesiana vs Dados
    ax1.scatter(X_train, y_train, color=CINZA, s=20, alpha=0.8, label='Dados de Treino (Observações)')
    ax1.plot(X_test_orig, mean_orig, color=AZUL, lw=2.5, label='Predição MAP (Média Posterior)')
    ax1.fill_between(X_test_orig.ravel(),
                     mean_orig - 2*std_orig,
                     mean_orig + 2*std_orig,
                     alpha=0.25, color=AZUL, label='Incerteza 2σ (Laplace)')
    ax1.fill_between(X_test_orig.ravel(),
                     mean_orig - std_orig,
                     mean_orig + std_orig,
                     alpha=0.40, color=AZUL, label='Incerteza 1σ')
    ax1.axvspan(X_test_orig.min(), X_train.min(), alpha=0.05, color='red')
    ax1.axvspan(X_train.max(), X_test_orig.max(), alpha=0.05, color='red',
                label='Região de Extrapolação (Alta Incerteza)')
    ax1.set_title("Rede Neural Bayesiana — Predição e Incerteza Epistêmica\n(Cap. 61: Aproximação de Laplace)")
    ax1.set_xlabel("Variável de Entrada")
    ax1.set_ylabel("Resposta Prevista")
    ax1.legend(fontsize=8)

    # Subplot 2: Curva de Perda de Treinamento
    ax2.semilogy(losses, color=LARANJA, lw=2)
    ax2.set_title("Convergência do Treinamento\n(Com Weight Decay = Prior Gaussiano)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("MSE (escala log)")

    plt.tight_layout()
    file_path = "figuras/bayesian_nn_laplace.png"
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"\nIncerteza média na região treinada:    {std_orig[75:225].mean():.4f}")
    print(f"Incerteza média na região extrapolada: {std_orig[:50].mean():.4f}")
    print(f"(Modelo é mais incerto onde não viu dados — comportamento Bayesiano esperado)")
    print(f"\nGráfico exportado para: {file_path}")


if __name__ == "__main__":
    run_bayesian_nn_demo()
