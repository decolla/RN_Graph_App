# Visualizador de Previsão de Séries Temporais (ONNX / WinForms)

Este projeto é uma aplicação Windows Forms (C#) desenvolvida para carregar dados históricos de séries temporais, executar inferência utilizando um modelo de Rede Neural pré-treinado (formato **ONNX**) e visualizar graficamente a comparação entre os dados reais e as previsões da IA.

O sistema foi projetado para trabalhar com modelos **KAN/Transformer** treinados em Python (PyTorch) e exportados para ONNX, replicando exatamente a normalização (**RobustScaler**) e o janelamento de dados utilizados no treinamento.

## 📋 Funcionalidades

* **Carregamento de Dados:** Leitura de arquivos `.csv` contendo dados históricos e features auxiliares.
* **Inferência ONNX:** Execução do modelo neural localmente via `Microsoft.ML.OnnxRuntime`.
* **Backtesting Completo:** Capacidade de percorrer todo o histórico do arquivo CSV gerando previsões em janelas deslizantes (Sliding Window) para criar uma linha contínua de previsão.
* **Visualização Gráfica:** Gráficos interativos (Zoom/Pan) utilizando **ZedGraph**.
* *Modo Simples:* Visualiza apenas uma janela de predição (336 steps).
* *Modo Histórico:* Visualiza a concatenação de todas as previsões ao longo do tempo (Eixo X em Data/Hora).


* **Suporte a Features Auxiliares:** Processamento de variáveis exógenas e *Time Embeddings* (Mês, Dia, Hora, Minuto) requeridos pelo modelo.
  

## ⚙️ Configurações do Modelo

As configurações abaixo estão *hardcoded* no código para garantir compatibilidade com o modelo `latest_config.yaml` fornecido:

* **Input Sequence Length (`seq_len`):** 720 steps (Histórico necessário para prever).
* **Prediction Length (`pred_len`):** 336 steps (Horizonte de previsão).
* **Features Principais:** 7 colunas (PEHIST, PSHIST, REGULADOR1, etc.).
* **Features Auxiliares:** 4 colunas.
* **Time Features:** 5 (Mês, Dia, DiaDaSemana, Hora, Minuto).


## 🛠 Detalhes Técnicos de Implementação

* **Normalização:** O software implementa `RobustScaler` (População) calculando Média e Desvio Padrão sobre todo o dataset carregado. Isso garante que os dados entrem na rede neural na mesma escala em que ela foi treinada.
* **Tratamento de Datas:** As datas são convertidas internamente para *Time Embeddings* normalizados entre -0.5 e 0.5, replicando a lógica da biblioteca `pandas` + `timefeatures` usada no Python.
* **Data de Corte:** A visualização está configurada para focar na previsão a partir de **11/10/2024**, garantindo que o gráfico não mostre histórico irrelevante.
