# Blind Test Pipeline

## 📁 Estrutura do Projeto

```
blind_tests/
├── data/                    # Datasets de entrada
│   ├── dataset_A.csv
│   ├── dataset_B.csv
│   ├── dataset_C.csv
│   ├── dataset_D.csv
│   ├── dataset_E.csv
│   └── dataset_F.csv
├── models/                  # Modelos pré-treinados
│   ├── model_1/
│   │   ├── model_1_FFNN.pkl
│   │   └── model_1_XGB.pkl
│   ├── model_2/
│   │   ├── model_2_FFNN.pkl
│   │   └── model_2_XGB.pkl
│   └── ... (model_3 até model_11)
├── results/                 # Resultados das predições
├── src/                     # Código fonte
└── 3_PREDICTIONS.py         # Script principal
```

## 🚀 Como Usar

### 1. Configuração Inicial

O script `3_PREDICTIONS.py` já está configurado com todos os modelos e datasets necessários. **Não é necessário modificar os dados ou modelos** - eles já estão nas pastas corretas.

### 2. Modificações Necessárias

Para executar o blind test, você precisa apenas modificar **DUAS LINHAS** no script:

```python
# USER CHANGE HERE
# Change the input data here (dataset_A, dataset_B, dataset_C, etc)
input_data_user = "data/dataset_B.csv"  # ← MUDAR AQUI
# Change the model folder here (model_1, model_2, model_3, etc)
model_folder = "model_2"  # ← MUDAR AQUI
```

#### Opções de Dataset:
- `"data/dataset_A.csv"`
- `"data/dataset_B.csv"`
- `"data/dataset_C.csv"`
- `"data/dataset_D.csv"`
- `"data/dataset_E.csv"`
- `"data/dataset_F.csv"`

#### Opções de Modelo:
- `"model_1"` até `"model_11"`

### 3. Execução

```bash
python 3_PREDICTIONS.py
```

## 📊 Saída

Os resultados são salvos automaticamente em:
```
results/{model_folder}/{dataset_name}/
├── predictions.csv          # Predições de todos os modelos compatíveis
├── evaluation_metrics.csv   # Métricas de avaliação
└── plots/                   # Gráficos de visualização
```

### Exemplo de Saída:
- Para `dataset_B` + `model_2`: `results/model_2/dataset_B/`

## 🔧 Funcionalidades

O script automaticamente:

1. **Carrega o dataset** especificado
2. **Identifica modelos compatíveis** baseado nas features disponíveis
3. **Faz predições** com todos os modelos compatíveis (FFNN e XGB)
4. **Avalia performance**
5. **Gera visualizações**:
   - Predicted vs Measured
   - Residuals
   - Predictions by combination
   - Baseline checks

## 📈 Features dos Modelos

Todos os modelos foram treinados com as mesmas features:
- `NPHI`
- `GR`
- `DT`
- `RHOB`

## ⚠️ Requisitos

Certifique-se de ter instaladas as dependências. Escolha uma das opções abaixo:

### Opção 1: Instalação com Python venv (Recomendado)
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows)
.venv\Scripts\activate
# Ativar ambiente (Linux/Mac)
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

**Nota:** Este script foi testado e validado com Python 3.11 e Python 3.12. Recomenda-se usar uma dessas versões para garantir compatibilidade total.

### Opção 2: Instalação com conda
```bash
# Criar ambiente conda (Python 3.11 ou 3.12)
conda create -n blind_test python=3.11
conda activate blind_test

# Instalar dependências
pip install -r requirements.txt
```

---

### Opção 3: Usar Docker e acessar o JupyterLab via navegador

1. Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/) no seu sistema.
2. No terminal, execute:
   ```bash
   docker-compose up --build
   ```
3. Acesse [http://localhost:8888/lab](http://localhost:8888/lab) no navegador.

---

### Opção 4: Usar Docker e acessar o container via VSCode (Remote - Containers)

1. Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/) e o [Visual Studio Code](https://code.visualstudio.com/).
2. Instale a extensão **Remote - Containers** no VSCode.
3. No terminal, execute:
   ```bash
   docker-compose up --build
   ```
   > **Importante:** O container precisa estar rodando para que o VSCode consiga conectar!
4. No VSCode, pressione `F1` e selecione `Remote-Containers: Attach to Running Container...` e escolha o container `blindtest`.
5. O VSCode abrirá um terminal e ambiente Python já configurado dentro do container, pronto para rodar scripts e notebooks.

---