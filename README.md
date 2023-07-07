
# Projeto de diagnóstico de doenças respiratórias usando imagens de raio-X de tórax e aprendizado de máquina

Este projeto usa diferentes modelos de aprendizado de máquina para rotular imagens de raio-X de tórax em quatro grupos: COVID-19, Normal, Pneumonia-Bacterial e Pneumonia-Viral. O projeto dá a opção de selecionar o classificador que se quer utilizar e realiza automaticamente o acerto dos hiper-parâmetros. Além disso, este projeto combina diversas técnicas de extração.

Os algoritmos utilizados são:

- **Histogram of Oriented Gradients (HOG)**, que é uma técnica de extração de características que registra a forma e a orientação dos objetos nas imagens;

- **Local Binary Pattern (LBP)**, que é uma técnica de extração de características que registra a textura e o contraste das imagens;

- **Hu Moments**, que são medidas de forma que são invariantes à escala, rotação e translação das imagens. Os classificadores utilizados são:

- **XGBoost**, que é um algoritmo de boosting baseado em árvores de decisão que otimiza o desempenho e a velocidade do aprendizado;

- **KNN**, que é um algoritmo de classificação baseado na proximidade entre os exemplos, usando uma medida de distância como critério;

- **SVM**, que é um algoritmo de classificação baseado na construção de um hiperplano ótimo que separa as classes com a maior margem possível;

- **MLP**, que é um algoritmo de classificação baseado em redes neurais artificiais com uma ou mais camadas ocultas.

- **Grid Search**, que é um algoritmo que testa todas as combinações possíveis de valores dos hiper-parâmetros e escolhe a melhor de acordo com uma métrica de avaliação.

Este projeto é o projeto final da disciplina de Processamento de Imagens UTFPR - Cornélio Procópio.


## Autor

- [@João Vitor da Costa Andrade](https://github.com/joaovcandrade)


## Dataset e Referência

 - [Dataset 3 kinds of Pneumonia](https://www.kaggle.com/datasets/artyomkolas/3-kinds-of-pneumonia?resource=download)
 - [Instruções de projeto final](https://github.com/matiassingers/awesome-readme)
 

## Funcionalidades

O projeto possui as seguintes funcionalidades:

- Carregar e processar as imagens do dataset;
- Codificar os rótulos das imagens em classes únicas;
- Extrair características HOG, textura e de forma das imagens;
- Dividir o dataset em conjuntos de treino e teste;
- Fazer o tunning e avaliar os modelos XGBoost, KNN, SVM e MLP usando Grid Search;
- Plotar a matriz de confusão e calcular a acurácia dos modelos.

## Instalação e execução

Para instalar as dependências do projeto, siga os passos abaixo:

Certifique-se de que você tem o Python 3.7 ou superior instalado na sua máquina. Você precisa ter o pip instalado. Você pode verificar a versão do Python e pip com o comando:

```bash
python --version
pip --version
```

Clone este repositório ou faça o download do código-fonte. Você pode clonar o repositório com o comando:

```bash
git clone https://github.com/fulano/projeto.git
```

Instale o dataset disponível neste [link](https://www.kaggle.com/datasets/artyomkolas/3-kinds-of-pneumonia?resource=download) e extraia seu conteúdo na pasta img. Seu diretório deverá se parecer com isto
```bash
img
├── COVID-19
├── Normal
├── Pneumonia-Bacterial
└── Pneumonia-Viral
```

Na raiz do projeto instale as dependências usando o pip ou pip3.
```bash
pip install -r requirements.txt
```

Isso irá instalar todos os pacotes necessários para executar o projeto.

Na raiz do projeto execute o scritp rec_rx.py utilizando python ou python3.
```bash
python rec_rx.py
```
O projeto é interativo e permite que você escolha qual modelo de aprendizado de máquina quer rodar. Basta digitar o nome do modelo (XGBoost, KNN, SVM ou MLP) quando solicitado. O programa irá fazer o tunning dos hiperparâmetros do modelo escolhido usando validação cruzada e depois irá avaliar o modelo com os dados de teste. Você verá um relatório de classificação com as principais métricas (precisão, recall, f1-score), uma matriz de confusão plotada e a acurácia do modelo.



## Stack utilizada

O projeto foi desenvolvido usando as seguintes tecnologias:

- Python 3.7
- OpenCV
- NumPy
- Scikit-image
- Scikit-learn
- XGBoost
- Seaborn
- Matplotlib


## Classificadores e Acurácia

- KNN: 0.66
- XGBoost 0.83
- SVM 0.78
- MLP: 0.75