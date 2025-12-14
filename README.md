# 📘 Podręcznik Inżynierii AI & Implementacje Referencyjne

Kompleksowy zbiór algorytmów zaimplementowanych od podstaw ("from first principles"), pokrywający pełne spektrum Inżynierii Uczenia Maszynowego: od Analizy Statystycznej i Klasycznego ML, aż po Duże Modele Językowe (LLM) i architektury Computer Vision.

### 🎯 Cel Repozytorium

Ten projekt służy jako **referencja techniczna** oraz baza wiedzy demonstrująca matematyczne fundamenty stojące za nowoczesnymi systemami AI. Wykracza poza używanie gotowych, wysokopoziomowych API, skupiając się na zrozumieniu tego, _jak_ i _dlaczego_ te algorytmy działają "pod maską".

### 🔑 Kluczowe Implementacje (Od Zera)

- **Architektura LLM:** Pełna implementacja Bloku Transformera (Self-Attention, LayerNorm, Residuals) w PyTorch + Positional Encoding + Flash Attention & KV Cache.
- **Optymalizacja:** Matematyczna implementacja LoRA (Low-Rank Adaptation) do fine-tuningu + Kwantyzacja (FP32 → INT8) + Product Quantization dla baz wektorowych.
- **Generative AI:** Sieci GAN, VAE (z wykorzystaniem Reparameterization Trick) oraz Diffusion Models (DDPM).
- **Computer Vision:** Ręczna implementacja mechanizmów IoU (Intersection over Union), NMS (Non-Max Suppression) oraz Vision Transformers (ViT).
- **ML Ops:** Niestandardowe Estymatory (Custom Estimators) i Pipeline'y Scikit-Learn do produkcyjnego przetwarzania danych.
- **Advanced ML:** Metric Learning (Siamese Networks), Graph Neural Networks, Contrastive Learning, Data Drift Detection.
- **Agenci AI:** LangChain ReAct, Prompt Engineering (CoT/ToT), RAG Evaluation, Speculative Decoding.
- **Next-Gen Architectures:** Mamba (State Space Models), Mixture of Experts (MoE), Liquid Neural Networks, Meta-Learning (MAML).

---

## 📂 Spis Treści i Tematyka

### 📊 Analiza Danych (EDA) i Statystyka

Fundamenty pracy z danymi. Jak zrozumieć, co siedzi w tabeli, zanim wrzucimy to do modelu.

| Plik                                                   | Temat                   | Kluczowe pojęcia                                                   |
| :----------------------------------------------------- | :---------------------- | :----------------------------------------------------------------- |
| **00_KDE_Tutorial.ipynb**                              | Rozkłady danych         | Kernel Density Estimation (KDE), histogramy, wizualizacja gęstości |
| **01_Correlations_and_Significance_of_Features.ipynb** | Badanie zależności      | Korelacja Pearsona/Spearmana, Heatmapy, p-value                    |
| **04_Statistics_and_Scaling.ipynb**                    | Testy hipotez i skala   | **P-value**, Test T-studenta, MinMaxScaler vs StandardScaler       |
| **35_ANOVA_Hypothesis_Testing.ipynb**                  | Porównywanie grup       | ANOVA, Test F, wariancja międzygrupowa                             |
| **42_Statistics_Masterclass.ipynb**                    | Statystyka zaawansowana | Testy parametryczne i nieparametryczne, rozkłady                   |

### 🛠️ Inżynieria Cech (Feature Engineering)

Przygotowanie "brudnych" danych, aby algorytmy mogły z nich korzystać.

| Plik                                    | Temat                    | Kluczowe pojęcia                                                        |
| :-------------------------------------- | :----------------------- | :---------------------------------------------------------------------- |
| **02_Advanced_Feature_Selection.ipynb** | Wybór najlepszych danych | RFE (Recursive Feature Elimination), SelectKBest, redukcja szumu        |
| **03_Encoding_Tutorial.ipynb**          | Zamiana tekstu na liczby | One-Hot Encoding, Label Encoding, Ordinal Encoding                      |
| **13_Missing_Data_Imputation.ipynb**    | Obsługa braków danych    | Imputacja (średnia, mediana), imputacja grupowa (Pandas transform), NaN |

### 🤖 Klasyczny Machine Learning

Algorytmy uczenia z nadzorem (Supervised) i bez nadzoru (Unsupervised).

| Plik                                          | Temat                    | Kluczowe pojęcia                                                      |
| :-------------------------------------------- | :----------------------- | :-------------------------------------------------------------------- |
| **06_Naive_Bayes_Spam.ipynb**                 | Filtr antyspamowy (NLP)  | **Naive Bayes**, Bag of Words, prawdopodobieństwo warunkowe           |
| **08_Overfitting_Underfitting.ipynb**         | Diagnoza błędów modelu   | Przeuczenie vs Niedouczenie, wielomiany, generalizacja                |
| **09_K_Means_Clustering.ipynb**               | Segmentacja klientów     | **Unsupervised Learning**, K-Means, Metoda Łokcia (Elbow Method)      |
| **10_Decision_Trees.ipynb**                   | Drzewa Decyzyjne         | White-Box Models, wizualizacja decyzji, Feature Importance            |
| **14_Random_Forest_Ensemble.ipynb**           | Ensemble Learning        | **Random Forest**, Bagging, agregacja predykcji, stabilność           |
| **19_Cross_Validation.ipynb**                 | Walidacja modeli         | **K-Fold**, walidacja krzyżowa, unikanie overfittingu                 |
| **27_Hyperparameter_Tuning_GridSearch.ipynb** | Optymalizacja modeli     | **Grid Search**, RandomizedSearch, dobór parametrów                   |
| **34_Regularization_Lasso_Ridge.ipynb**       | Regularyzacja            | **Lasso (L1)**, Ridge (L2), ElasticNet, kara za złożoność             |
| **36_Market_Basket_Apriori.ipynb**            | Analiza Koszykowa        | **Apriori**, Support, Confidence, Lift, reguły asocjacyjne            |
| **37_Gradient_Boosting_XGBoost.ipynb**        | Gradient Boosting        | **XGBoost**, LightGBM, uczenie sekwencyjne, boosting                  |
| **47_SVM_Kernel_Trick.ipynb**                 | Support Vector Machines  | **SVM**, Kernel Trick, hyperplanes, separowalność liniowa             |
| **51_Recommender_Systems_SVD.ipynb**          | Systemy Rekomendacyjne   | **SVD**, Matrix Factorization, collaborative filtering                |
| **60_Bayesian_Optimization_Optuna.ipynb**     | Optymalizacja Bayesowska | **Optuna**, Bayesian Optimization, inteligentny dobór hiperparametrów |

### 📏 Ewaluacja Modeli

Jak sprawdzić, czy model naprawdę działa?

| Plik                                           | Temat           | Kluczowe pojęcia                                                            |
| :--------------------------------------------- | :-------------- | :-------------------------------------------------------------------------- |
| **07_Confusion_Matrix_Precision_Recall.ipynb** | Metryki sukcesu | **Macierz Pomyłek**, Precision, Recall, F1-Score (dlaczego Accuracy kłamie) |

### 🧠 Fundamenty LLM i Generative AI

Mechanizmy stojące za modelami takimi jak GPT.

| Plik                                         | Temat                          | Kluczowe pojęcia                                                                      |
| :------------------------------------------- | :----------------------------- | :------------------------------------------------------------------------------------ |
| **05_Top_p_Top_k.ipynb**                     | Sterowanie generowaniem tekstu | Sampling, probabilistyka wyboru słów, kreatywność AI                                  |
| **11_Embeddings_Vector_Space.ipynb**         | Wektory słów                   | **Embeddings**, przestrzeń wektorowa, algebra na słowach (Król - Mężczyzna + Kobieta) |
| **12_LLM_Temperature.ipynb**                 | Parametr Temperatury           | Softmax, Logits, sterowanie halucynacjami i pewnością modelu                          |
| **23_Tokenization_GPT.ipynb**                | Tokenizacja                    | **Byte Pair Encoding**, subword tokenization, problem z liczeniem liter               |
| **24_Self_Attention_Mechanism.ipynb**        | Mechanizm Uwagi                | **Transformer**, Query-Key-Value, kontekst w zdaniach                                 |
| **18_Cosine_Similarity_Search.ipynb**        | Podobieństwo wektorów          | **Cosine Similarity**, kąt vs odległość, Semantic Search                              |
| **20_RAG_Architecture_Simulation.ipynb**     | Retrieval Augmented Generation | **RAG**, wyszukiwanie w bazie wiedzy, pipeline z embeddingami                         |
| **26_RAG_Chunking_Strategies.ipynb**         | Przygotowanie dokumentów       | **Chunking**, Fixed-size, Recursive, Overlap, Windowing                               |
| **46_Transformer_Block_From_Scratch.ipynb**  | Blok Transformera              | **Transformer Block**, LayerNorm, Residual Connections, Feed Forward                  |
| **55_LoRA_Fine_Tuning_Math.ipynb**           | Fine-tuning LLM                | **LoRA**, Low-Rank Adaptation, efektywne douczanie modeli                             |
| **56_Positional_Encoding_Transformer.ipynb** | GPS Transformera               | **Positional Encoding**, sinusy i cosinusy, kolejność w sekwencjach                   |
| **64_Knowledge_Distillation.ipynb**          | Kompresja modeli               | **Teacher-Student**, Soft Labels, Temperature, transfer wiedzy                        |
| **68_RLHF_PPO_ChatGPT_Alignment.ipynb**      | Alignment LLM                  | **PPO**, RLHF, uczenie przez feedback ludzki, jak powstał ChatGPT                     |

### 🚀 Optymalizacja LLM i Next-Gen Architectures

Nowoczesne architektury i techniki optymalizacji dla produkcyjnych systemów AI.

| Plik                                    | Temat                      | Kluczowe pojęcia                                                           |
| :-------------------------------------- | :------------------------- | :------------------------------------------------------------------------- |
| **71_LLM_Optimization_KV_Cache.ipynb**  | Flash Attention & KV Cache | **KV Cache**, Flash Attention, Tiling, optymalizacja O(N²), pamięć GPU     |
| **72_Mamba_State_Space_Models.ipynb**   | State Space Models         | **Mamba**, SSM, dyskretyzacja równań różniczkowych, złożoność liniowa      |
| **73_Mixture_of_Experts_MoE.ipynb**     | Mixture of Experts         | **MoE**, Gating Network, Router, architektura GPT-4, sparse models         |
| **74_Liquid_Neural_Networks_LFC.ipynb** | Liquid Neural Networks     | **LFC**, adaptive weights, równania różniczkowe, robotyka, drony           |
| **75_Meta_Learning_MAML.ipynb**         | Meta-Learning              | **MAML**, Model-Agnostic Meta-Learning, few-shot learning, fast adaptation |

### 🤖 Agenci AI i LLM Engineering

Najgorętszy temat 2025 roku. AI, które "działa", a nie tylko "gada".

| Plik                                                 | Temat                        | Kluczowe pojęcia                                                        |
| :--------------------------------------------------- | :--------------------------- | :---------------------------------------------------------------------- |
| **76_LangChain_ReAct_Agent.ipynb**                   | Agenci AI                    | **ReAct**, Reason+Act, pętla agenta, narzędzia, akcje                   |
| **77_Prompt_Engineering_CoT_ToT.ipynb**              | Prompt Engineering           | **Chain of Thought**, Tree of Thoughts, reasoning, myślenie na głos     |
| **78_RAG_Evaluation_RAGAS.ipynb**                    | Ewaluacja RAG                | **RAGAS**, Faithfulness, Answer Relevance, metryki jakości RAG          |
| **79_Vector_Compression_Product_Quantization.ipynb** | Vector Database Optimization | **Product Quantization**, IVF-PQ, FAISS, kompresja wektorów, skalowanie |
| **80_Speculative_Decoding.ipynb**                    | Przyspieszanie Inferencji    | **Speculative Decoding**, draft model, verification, 2-3x speedup       |

### 🧮 Matematyka i Optymalizacja

Jak maszyny się uczą pod maską?

| Plik                                              | Temat               | Kluczowe pojęcia                                                      |
| :------------------------------------------------ | :------------------ | :-------------------------------------------------------------------- |
| **15_Gradient_Descent.ipynb**                     | Optymalizacja       | **Gradient Descent**, Learning Rate, schodzenie po gradiencie         |
| **17_PCA_Dimensionality_Reduction.ipynb**         | Redukcja wymiarów   | **PCA**, Principal Component Analysis, wizualizacja wysokich wymiarów |
| **48_tSNE_vs_PCA_Dimensionality_Reduction.ipynb** | Redukcja nieliniowa | **t-SNE**, UMAP, wizualizacja embeddingów                             |

### 🔬 Sieci Neuronowe i Deep Learning

Od pojedynczego neuronu do głębokich sieci.

| Plik                                            | Temat                           | Kluczowe pojęcia                                                 |
| :---------------------------------------------- | :------------------------------ | :--------------------------------------------------------------- |
| **16_Neural_Network_Perceptron.ipynb**          | Pierwszy neuron                 | **Perceptron**, wagi, bias, funkcja aktywacji                    |
| **21_MLP_Neural_Network_XOR.ipynb**             | Sieci wielowarstwowe            | **Multi-Layer Perceptron**, warstwy ukryte, XOR problem          |
| **22_Activation_Functions.ipynb**               | Funkcje aktywacji               | **ReLU**, Sigmoid, Softmax, nieliniowość                         |
| **32_PyTorch_Tensors_Autograd.ipynb**           | Podstawy PyTorch                | **Tensors**, Autograd, automatyczne różniczkowanie               |
| **33_PyTorch_Neural_Network_Class.ipynb**       | Budowa sieci w PyTorch          | **nn.Module**, forward pass, OOP w deep learningu                |
| **38_CNN_Computer_Vision.ipynb**                | Sieci Konwolucyjne              | **CNN**, Conv2d, MaxPool, filtry, Computer Vision                |
| **39_RNN_LSTM_Sequence_Models.ipynb**           | Sieci Rekurencyjne              | **RNN**, LSTM, przetwarzanie sekwencji, pamięć                   |
| **40_Autoencoder_Anomaly_Detection.ipynb**      | Detekcja Anomalii               | **Autoencoder**, kompresja, detekcja outlierów                   |
| **41_GAN_Generative_Adversarial_Network.ipynb** | Generative Adversarial Networks | **GAN**, Generator, Dyskryminator, generowanie danych            |
| **43_VAE_Variational_Autoencoder.ipynb**        | Variational Autoencoder         | **VAE**, Latent Space, KL Divergence, generowanie                |
| **49_Object_Detection_IoU.ipynb**               | Detekcja Obiektów               | **IoU**, Intersection over Union, bounding boxes                 |
| **50_UNet_Image_Segmentation.ipynb**            | Segmentacja Obrazu              | **U-Net**, segmentacja pikselowa, architektura U                 |
| **61_Normalization_Layers_BN_vs_LN.ipynb**      | Warstwy Normalizacji            | **Batch Norm**, Layer Norm, Instance Norm, stabilizacja treningu |
| **70_Vision_Transformer_ViT.ipynb**             | Vision Transformers             | **ViT**, Patches, koniec ery CNN, Self-Attention w obrazach      |

### 🎮 Reinforcement Learning

Uczenie przez nagrody i kary.

| Plik                                  | Temat           | Kluczowe pojęcia                               |
| :------------------------------------ | :-------------- | :--------------------------------------------- |
| **44_RL_Q_Learning_FrozenLake.ipynb** | Q-Learning      | **Q-Table**, Równanie Bellmana, nagrody i kary |
| **45_RL_Deep_Q_Learning_DQN.ipynb**   | Deep Q-Learning | **DQN**, Replay Buffer, sieci neuronowe w RL   |

### 🧬 Algorytmy Zaawansowane

Specjalistyczne techniki i podejścia.

| Plik                                       | Temat                  | Kluczowe pojęcia                                                     |
| :----------------------------------------- | :--------------------- | :------------------------------------------------------------------- |
| **52_Genetic_Algorithms_Evolution.ipynb**  | Algorytmy Genetyczne   | **Evolutionary Algorithms**, krzyżowanie, mutacja, selekcja          |
| **53_Monte_Carlo_Simulation.ipynb**        | Symulacje Monte Carlo  | Symulacje probabilistyczne, analiza ryzyka                           |
| **54_FFT_Signal_Processing.ipynb**         | Przetwarzanie Sygnałów | **FFT**, Transformata Fouriera, analiza częstotliwości               |
| **59_Model_Quantization_INT8.ipynb**       | Kwantyzacja Modeli     | **Quantization**, FP32→INT8, kompresja, odpalanie AI na edge devices |
| **62_Time_Series_Decomposition_STL.ipynb** | Dekompozycja Szeregów  | **STL**, Trend, Sezonowość, Reszta, analiza biznesowa                |

### 🎨 Generative AI - Zaawansowane

Modele generatywne nowej generacji.

| Plik                                     | Temat                | Kluczowe pojęcia                                                 |
| :--------------------------------------- | :------------------- | :--------------------------------------------------------------- |
| **63_Diffusion_Models_DDPM.ipynb**       | Diffusion Models     | **DDPM**, Forward/Reverse Diffusion, matematyka Stable Diffusion |
| **67_Contrastive_Learning_SimCLR.ipynb** | Contrastive Learning | **SimCLR**, uczenie kontrastowe, Self-Supervised Learning        |

### 🕸️ Graph Neural Networks

Dane w formie grafów i relacji.

| Plik                                   | Temat                 | Kluczowe pojęcia                                                 |
| :------------------------------------- | :-------------------- | :--------------------------------------------------------------- |
| **58_Graph_Neural_Networks_GNN.ipynb** | Graph Neural Networks | **GNN**, Message Passing, macierze przyległości, sieci społeczne |

### 🔍 Metric Learning & Similarity

Uczenie odległości i podobieństwa.

| Plik                                 | Temat            | Kluczowe pojęcia                                                  |
| :----------------------------------- | :--------------- | :---------------------------------------------------------------- |
| **57_Metric_Learning_Siamese.ipynb** | Siamese Networks | **Triplet Loss**, Metric Learning, FaceID, weryfikacja tożsamości |

### 🔧 Vector Search & Optimization

Efektywne wyszukiwanie w wysokich wymiarach.

| Plik                                   | Temat              | Kluczowe pojęcia                                             |
| :------------------------------------- | :----------------- | :----------------------------------------------------------- |
| **65_HNSW_Vector_Search_Engine.ipynb** | Vector Search      | **HNSW**, Hierarchical Navigable Small World, bazy wektorowe |
| **66_Kalman_Filter_Tracking.ipynb**    | Śledzenie Obiektów | **Kalman Filter**, filtracja predykcyjna, GPS, robotyka      |

### 📈 MLOps & Production

Monitoring i wdrożenia produkcyjne.

| Plik                                         | Temat                  | Kluczowe pojęcia                                               |
| :------------------------------------------- | :--------------------- | :------------------------------------------------------------- |
| **25_Model_Persistence_Pickle_Joblib.ipynb** | Zapisywanie modeli     | **Pickle**, Joblib, serializacja obiektów                      |
| **30_Sklearn_Pipelines.ipynb**               | Rurociągi ML           | **Pipeline**, StandardScaler, data leakage prevention          |
| **31_Custom_Transformers.ipynb**             | Własne transformery    | **BaseEstimator**, TransformerMixin, fit-transform pattern     |
| **28_Python_Dataclasses_for_ML.ipynb**       | Konfiguracja modeli    | **Dataclasses**, structured configs, TrainingArguments pattern |
| **29_OOP_Classmethod_Staticmethod.ipynb**    | Wzorce projektowe      | **@classmethod**, @staticmethod, ModelLoader, factory pattern  |
| **69_Data_Drift_Detection_PSI.ipynb**        | Monitoring Produkcyjny | **Data Drift**, KS-Test, PSI, wykrywanie zmian w danych        |

---

## 🛠️ Technologie

Projekt oparty na standardowym stacku Data Science:

- **Python 3.x**
- **Pandas & NumPy** (Manipulacja danymi i obliczenia)
- **Scikit-Learn** (Algorytmy ML, Preprocessing, Metryki)
- **Matplotlib & Seaborn** (Wizualizacja danych)
- **SciPy** (Testy statystyczne)
- **PyTorch** (Deep Learning Framework)
- **Optuna** (Bayesian Optimization)

## 🚀 Jak używać tego podręcznika?

Masz dwie możliwości uruchomienia kodu: szybką (w chmurze) i profesjonalną (lokalnie).

### ☁️ Opcja 1: Google Colab (Bez instalacji)

Najszybszy sposób na naukę. Każdy notatnik w tym repozytorium posiada przycisk **"Open in Colab"** na samej górze.

1.  Otwórz wybrany plik `.ipynb` na liście plików.
2.  Kliknij przycisk <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle">.
3.  Kod uruchomi się natychmiast na darmowych GPU od Google.

### 💻 Opcja 2: Lokalnie (VS Code + uv)

Zalecane dla inżynierów budujących własne środowisko.

1.  **Sklonuj repozytorium:**

    ```bash
    git clone https://github.com/takzen/ai-engineering-handbook.git
    cd ai-engineering-handbook
    ```

2.  **Stwórz i aktywuj środowisko wirtualne:**

    ```bash
    uv venv

    # Windows:
    .\.venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Zainstaluj zależności (PyTorch + ML Stack):**

    ```bash
    # 1. PyTorch (Wersja z obsługą CUDA: najnowsza stabilna (12.12.2025) jest cu130)
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

    # 2. Reszta narzędzi (Pandas, Scikit-Learn, SHAP, etc.)
    uv pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels shap xgboost mlxtend gym gymnasium notebook ipykernel networkx optuna ipywidgets plotly
    ```

    _(Uwaga: Wersję `cu130` w linku PyTorcha możesz dostosować do sterowników swojej karty graficznej)._

---

## 📊 Statystyki Projektu

- **80 notatników** pokrywających pełne spektrum AI/ML
- **Od podstaw matematycznych** do produkcyjnych implementacji
- **Ponad 25 kategorii tematycznych** (EDA, Classical ML, Deep Learning, LLM, Computer Vision, RL, Agenci AI, MLOps)
- **Implementacje referencyjne** algorytmów używanych w produkcji (Transformers, Diffusion, HNSW, Kalman, PPO, Mamba, MoE)
- **Najnowsze architektury 2024/2025:** Flash Attention, Mamba SSM, Mixture of Experts, Liquid Networks, Meta-Learning

---

Autor: Krzysztof Pika
