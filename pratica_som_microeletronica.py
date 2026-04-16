"""
Mapas Auto-Organizáveis (SOM) de Kohonen
Baseado no Cap. 67 do livro 'Aprendizado de Máquina'.

Este script implementa o algoritmo SOM do zero para visualização e 
agrupamento de dados de alta dimensão, preservando a topologia original 
em uma grade 2D.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações estéticas
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": False
})

class SOMScratch:
    """Implementaçao manual de Mapas Auto-Organizáveis (Kohonen)."""
    
    def __init__(self, grid_size=(10, 10), input_dim=3, alpha=0.5, sigma=None):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.alpha_init = alpha
        self.sigma_init = sigma if sigma else max(grid_size) / 2.0
        
        # Inicializar pesos aleatoriamente
        self.weights = np.random.uniform(0, 1, (grid_size[0], grid_size[1], input_dim))
        
        # Criar grid de coordenadas para cálculo de vizinhança
        self.grid_x, self.grid_y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), indexing='ij')

    def _find_bmu(self, x):
        """Encontra a Best Matching Unit (BMU) para o vetor x."""
        # Distância Euclidiana entre x e todos os pesos
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Índice do neurônio com menor distância
        return np.unravel_index(np.argmin(distances), self.grid_size)

    def _update_weights(self, x, bmu_idx, it, total_it):
        """Atualiza os pesos do vencedor e vizinhos."""
        # Decaimento paramétrico
        alpha = self.alpha_init * (1 - it / total_it)
        sigma = self.sigma_init * (1 - it / total_it)
        
        # Distância quadrada na grade até a BMU
        dist_sq = (self.grid_x - bmu_idx[0])**2 + (self.grid_y - bmu_idx[1])**2
        
        # Função de vizinhança Gaussiana
        influence = np.exp(-dist_sq / (2 * sigma**2 + 1e-9))
        
        # Atualização: w = w + alpha * influence * (x - w)
        self.weights += alpha * influence[:, :, np.newaxis] * (x - self.weights)

    def fit(self, data, epochs=100):
        N = data.shape[0]
        total_it = epochs * N
        
        print(f"Treinando SOM {self.grid_size[0]}x{self.grid_size[1]} ({epochs} épocas)...")
        
        it = 0
        for epoch in range(epochs):
            # Embaralhar dados a cada época
            np.random.shuffle(data)
            for x in data:
                bmu = self._find_bmu(x)
                self._update_weights(x, bmu, it, total_it)
                it += 1
        return self

# =============================================================================
# EXPERIMENTO: ANÁLISE DE DEFEITOS EM WAFERS (CLUSTERING DE SENSORES)
# =============================================================================

def run_som_experiment():
    np.random.seed(42)
    # Simulamos 150 wafers com dados de 5 sensores (Pressão, Temp, Fluxo, Tensão, Umidade)
    n_samples = 150
    input_dim = 5
    
    # Criar 3 tipos de estados de processo:
    # 1. Saudável (Normal)
    healthy = np.random.normal(0.2, 0.05, (50, input_dim))
    # 2. Defeito de Anel (Alta temperatura/pressão local)
    defect_ring = np.random.normal(0.7, 0.1, (50, input_dim))
    # 3. Defeito de Borda (Problema de fluxo na borda do wafer)
    defect_edge = np.random.normal(0.4, 0.05, (50, input_dim))
    defect_edge[:, 2] += 0.4 # Sensor de fluxo alterado
    
    data = np.vstack([healthy, defect_ring, defect_edge])
    # Normalizar 0-1
    data = (data - data.min()) / (data.max() - data.min())
    
    # Ajustar SOM
    som = SOMScratch(grid_size=(15, 15), input_dim=input_dim).fit(data, epochs=200)
    
    # VISUALIZAÇÃO
    os.makedirs("figuras", exist_ok=True)
    
    # Mapa de Ativação (BMU para cada dado)
    mapping = np.zeros((15, 15))
    for i, x in enumerate(data):
        bmu = som._find_bmu(x)
        if i < 50: mapping[bmu] = 1 # Saudável
        elif i < 100: mapping[bmu] = 2 # Anel
        else: mapping[bmu] = 3 # Borda
        
    plt.figure(figsize=(8, 8))
    plt.imshow(mapping, cmap='viridis', interpolation='nearest')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Estado do Processo')
    plt.clim(-0.5, 3.5)
    
    # Legenda customizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Sem Ativação', markerfacecolor='#440154', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Saudável', markerfacecolor='#31688e', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Defeito de Anel', markerfacecolor='#35b779', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Defeito de Borda', markerfacecolor='#fde725', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1.0))
    
    plt.title("SOM: Mapa de Auto-Organização de Defeitos em Wafers\n(Agrupamento em Grade 2D)")
    plt.savefig("figuras/som_wafer_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResultados:")
    print(f"Mapa gerado e salvo em 'figuras/som_wafer_results.png'")

if __name__ == "__main__":
    run_som_experiment()
