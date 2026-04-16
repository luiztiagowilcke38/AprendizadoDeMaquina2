"""
Visão Computacional: Redes Neurais Convolucionais (CNN)
Baseado no Cap. 09 do livro 'Aprendizado de Máquina' 

Este script demonstra a arquitetura de uma CNN para reconhecimento de dígitos (MNIST),
incluindo visualização de filtros (kernels) e mapas de características.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Configurações estéticas profissionais (Seguindo o padrão do livro)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3
})

AZUL = '#1f77b4'
VERDE = '#2ca02c'

# 1. ARQUITETURA DA REDE (Conforme apresentado no Laboratório do Cap. 09)
class MiniCNN(nn.Module):
    """
    Arquitetura Mini-CNN para MNIST:
    2 Camadas Convolucionais + 2 Camadas Densas.
    """
    def __init__(self):
        super(MiniCNN, self).__init__()
        # Camada 1: 1 canal de entrada (tons de cinza) -> 16 filtros 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camada 2: 16 canais -> 32 filtros 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # O MNIST tem 28x28. 
        # Após conv1 + pool1: (28+2)/2 = 14x14
        # Após conv2 + pool2: (14+2)/2 = 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 classes para os dígitos 0-9

    def forward(self, x):
        # Primeira ativação convolucional
        x = self.pool(F.relu(self.conv1(x)))
        # Segunda ativação convolucional
        x = self.pool(F.relu(self.conv2(x)))
        
        # Achatar (Flatten) para as camadas densas
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# 2. FUNÇÃO DE TREINAMENTO
def train_epoch(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    return total_loss / len(train_loader)

# 3. VISUALIZAÇÕES E DIAGNÓSTICO
def generate_educational_plots(model, test_loader, device):
    model.eval()
    os.makedirs("figuras", exist_ok=True)
    
    # Selecionar uma amostra para visualização
    data, target = next(iter(test_loader))
    sample_img = data[0].unsqueeze(0).to(device)
    
    print("\nGerando figuras didáticas...")

    # A. Visualizar Kernels da Camada Conv1
    # Cada kernel aprende um detector de bordas ou texturas
    kernels = model.conv1.weight.detach().cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(kernels[i, 0], cmap='viridis')
        ax.axis('off')
        ax.set_title(f"F{i+1}", fontsize=9)
    plt.suptitle("Kernels da Camada Convolucional 1 (Filtros Aprendidos)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figuras/cnn_kernels.png", dpi=300)
    plt.close()

    # B. Visualizar Mapas de Características (Ativações)
    # Mostra como a imagem é "percebida" por cada filtro
    with torch.no_grad():
        activations = F.relu(model.conv1(sample_img))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(activations[0, i].cpu().numpy(), cmap='magma')
        ax.axis('off')
    plt.suptitle("Mapas de Características (Feature Maps - Camada 1)")
    plt.savefig("figuras/cnn_feature_maps.png", dpi=300)
    plt.close()

    # C. Exemplo de Predição com Probabilidades
    output = model(sample_img)
    probs = torch.exp(output).cpu().detach().numpy()[0]
    pred = np.argmax(probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(sample_img[0, 0].cpu().numpy(), cmap='gray')
    ax1.set_title(f"Imagem de Entrada (Alvo: {target[0]})")
    ax1.axis('off')
    
    ax2.bar(range(10), probs, color=AZUL, alpha=0.7)
    ax2.set_xticks(range(10))
    ax2.set_title(f"Distribuição de Probabilidade (Predição: {pred})")
    ax2.set_xlabel("Dígito")
    ax2.set_ylabel("Confiança")
    
    plt.tight_layout()
    plt.savefig("figuras/cnn_prediction_analysis.png", dpi=300)
    plt.close()

def main():
    # Detectar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformações: Tensor + Normalização (parâmetros clássicos do MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Baixando/Carregando dataset MNIST...")
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Inicializar modelo e otimizador
    model = MiniCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Treinamento rápido (1 época para demonstração didática)
    print("\nIniciando Ciclo de Aprendizado...")
    train_epoch(model, train_loader, optimizer, 1, device)
    
    # Geração de resultados
    generate_educational_plots(model, test_loader, device)
    
    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("Figuras salvas em 'figuras/':")
    print("1. cnn_kernels.png          - Os filtros espaciais aprendidos")
    print("2. cnn_feature_maps.png     - As ativações internas da rede")
    print("3. cnn_prediction_analysis.png - Análise de predição de um dígito")
    print("="*50)

if __name__ == "__main__":
    main()
