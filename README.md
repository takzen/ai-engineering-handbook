# 📘 Podręcznik Inżynierii AI & Implementacje Referencyjne

Kompleksowy zbiór algorytmów zaimplementowanych od podstaw ("from first principles"), pokrywający pełne spektrum Inżynierii Uczenia Maszynowego — od Analizy Statystycznej i Klasycznego ML, aż po Duże Modele Językowe (LLM) i architektury Computer Vision.

### 🎯 Cel Repozytorium

Ten projekt służy jako **referencja techniczna** oraz baza wiedzy demonstrująca matematyczne fundamenty stojące za nowoczesnymi systemami AI. Wykracza poza używanie gotowych, wysokopoziomowych API, skupiając się na zrozumieniu tego, _jak_ i _dlaczego_ te algorytmy działają "pod maską".

### 🔑 Kluczowe Implementacje (Od Zera)

- **Architektura LLM:** Pełna implementacja Bloku Transformera (Self-Attention, LayerNorm, Residuals) w PyTorch.
- **Optymalizacja:** Matematyczna implementacja LoRA (Low-Rank Adaptation) do fine-tuningu.
- **Generative AI:** Sieci GAN i VAE (z wykorzystaniem Reparameterization Trick).
- **Computer Vision:** Ręczna implementacja mechanizmów IoU (Intersection over Union) oraz NMS (Non-Max Suppression) dla detekcji obiektów.
- **ML Ops:** Niestandardowe Estymatory (Custom Estimators) i Pipeline'y Scikit-Learn do produkcyjnego przetwarzania danych.

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

| Plik                                          | Temat                   | Kluczowe pojęcia                                                 |
| :-------------------------------------------- | :---------------------- | :--------------------------------------------------------------- |
| **06_Naive_Bayes_Spam.ipynb**                 | Filtr antyspamowy (NLP) | **Naive Bayes**, Bag of Words, prawdopodobieństwo warunkowe      |
| **08_Overfitting_Underfitting.ipynb**         | Diagnoza błędów modelu  | Przeuczenie vs Niedouczenie, wielomiany, generalizacja           |
| **09_K_Means_Clustering.ipynb**               | Segmentacja klientów    | **Unsupervised Learning**, K-Means, Metoda Łokcia (Elbow Method) |
| **10_Decision_Trees.ipynb**                   | Drzewa Decyzyjne        | White-Box Models, wizualizacja decyzji, Feature Importance       |
| **14_Random_Forest_Ensemble.ipynb**           | Ensemble Learning       | **Random Forest**, Bagging, agregacja predykcji, stabilność      |
| **19_Cross_Validation.ipynb**                 | Walidacja modeli        | **K-Fold**, walidacja krzyżowa, unikanie overfittingu            |
| **27_Hyperparameter_Tuning_GridSearch.ipynb** | Optymalizacja modeli    | **Grid Search**, RandomizedSearch, dobór parametrów              |
| **34_Regularization_Lasso_Ridge.ipynb**       | Regularyzacja           | **Lasso (L1)**, Ridge (L2), ElasticNet, kara za złożoność        |
| **36_Market_Basket_Apriori.ipynb**            | Analiza Koszykowa       | **Apriori**, Support, Confidence, Lift, reguły asocjacyjne       |
| **37_Gradient_Boosting_XGBoost.ipynb**        | Gradient Boosting       | **XGBoost**, LightGBM, uczenie sekwencyjne, boosting             |
| **47_SVM_Kernel_Trick.ipynb**                 | Support Vector Machines | **SVM**, Kernel Trick, hyperplanes, separowalność liniowa        |
| **51_Recommender_Systems_SVD.ipynb**          | Systemy Rekomendacyjne  | **SVD**, Matrix Factorization, collaborative filtering           |

### 📏 Ewaluacja Modeli

Jak sprawdzić, czy model naprawdę działa?

| Plik                                           | Temat           | Kluczowe pojęcia                                                            |
| :--------------------------------------------- | :-------------- | :-------------------------------------------------------------------------- |
| **07_Confusion_Matrix_Precision_Recall.ipynb** | Metryki sukcesu | **Macierz Pomyłek**, Precision, Recall, F1-Score (dlaczego Accuracy kłamie) |

### 🧠 Fundamenty LLM i Generative AI

Mechanizmy stojące za modelami takimi jak GPT.

| Plik                                        | Temat                          | Kluczowe pojęcia                                                                      |
| :------------------------------------------ | :----------------------------- | :------------------------------------------------------------------------------------ |
| **05_Top_p_Top_k.ipynb**                    | Sterowanie generowaniem tekstu | Sampling, probabilistyka wyboru słów, kreatywność AI                                  |
| **11_Embeddings_Vector_Space.ipynb**        | Wektory słów                   | **Embeddings**, przestrzeń wektorowa, algebra na słowach (Król - Mężczyzna + Kobieta) |
| **12_LLM_Temperature.ipynb**                | Parametr Temperatury           | Softmax, Logits, sterowanie halucynacjami i pewnością modelu                          |
| **23_Tokenization_GPT.ipynb**               | Tokenizacja                    | **Byte Pair Encoding**, subword tokenization, problem z liczeniem liter               |
| **24_Self_Attention_Mechanism.ipynb**       | Mechanizm Uwagi                | **Transformer**, Query-Key-Value, kontekst w zdaniach                                 |
| **18_Cosine_Similarity_Search.ipynb**       | Podobieństwo wektorów          | **Cosine Similarity**, kąt vs odległość, Semantic Search                              |
| **20_RAG_Architecture_Simulation.ipynb**    | Retrieval Augmented Generation | **RAG**, wyszukiwanie w bazie wiedzy, pipeline z embeddingami                         |
| **26_RAG_Chunking_Strategies.ipynb**        | Przygotowanie dokumentów       | **Chunking**, Fixed-size, Recursive, Overlap, Windowing                               |
| **46_Transformer_Block_From_Scratch.ipynb** | Blok Transformera              | **Transformer Block**, LayerNorm, Residual Connections, Feed Forward                  |
| **55_LoRA_Fine_Tuning_Math.ipynb**          | Fine-tuning LLM                | **LoRA**, Low-Rank Adaptation, efektywne douczanie modeli                             |

### 🧮 Matematyka i Optymalizacja

Jak maszyny się uczą pod maską?

| Plik                                              | Temat               | Kluczowe pojęcia                                                      |
| :------------------------------------------------ | :------------------ | :-------------------------------------------------------------------- |
| **15_Gradient_Descent.ipynb**                     | Optymalizacja       | **Gradient Descent**, Learning Rate, schodzenie po gradiencie         |
| **17_PCA_Dimensionality_Reduction.ipynb**         | Redukcja wymiarów   | **PCA**, Principal Component Analysis, wizualizacja wysokich wymiarów |
| **48_tSNE_vs_PCA_Dimensionality_Reduction.ipynb** | Redukcja nieliniowa | **t-SNE**, UMAP, wizualizacja embeddingów                             |

### 🔬 Sieci Neuronowe i Deep Learning

Od pojedynczego neuronu do głębokich sieci.

| Plik                                            | Temat                           | Kluczowe pojęcia                                        |
| :---------------------------------------------- | :------------------------------ | :------------------------------------------------------ |
| **16_Neural_Network_Perceptron.ipynb**          | Pierwszy neuron                 | **Perceptron**, wagi, bias, funkcja aktywacji           |
| **21_MLP_Neural_Network_XOR.ipynb**             | Sieci wielowarstwowe            | **Multi-Layer Perceptron**, warstwy ukryte, XOR problem |
| **22_Activation_Functions.ipynb**               | Funkcje aktywacji               | **ReLU**, Sigmoid, Softmax, nieliniowość                |
| **32_PyTorch_Tensors_Autograd.ipynb**           | Podstawy PyTorch                | **Tensors**, Autograd, automatyczne różniczkowanie      |
| **33_PyTorch_Neural_Network_Class.ipynb**       | Budowa sieci w PyTorch          | **nn.Module**, forward pass, OOP w deep learningu       |
| **38_CNN_Computer_Vision.ipynb**                | Sieci Konwolucyjne              | **CNN**, Conv2d, MaxPool, filtry, Computer Vision       |
| **39_RNN_LSTM_Sequence_Models.ipynb**           | Sieci Rekurencyjne              | **RNN**, LSTM, przetwarzanie sekwencji, pamięć          |
| **40_Autoencoder_Anomaly_Detection.ipynb**      | Detekcja Anomalii               | **Autoencoder**, kompresja, detekcja outlierów          |
| **41_GAN_Generative_Adversarial_Network.ipynb** | Generative Adversarial Networks | **GAN**, Generator, Dyskryminator, generowanie danych   |
| **43_VAE_Variational_Autoencoder.ipynb**        | Variational Autoencoder         | **VAE**, Latent Space, KL Divergence, generowanie       |
| **49_Object_Detection_IoU.ipynb**               | Detekcja Obiektów               | **IoU**, Intersection over Union, bounding boxes        |
| **50_UNet_Image_Segmentation.ipynb**            | Segmentacja Obrazu              | **U-Net**, segmentacja pikselowa, architektura U        |

### 🎮 Reinforcement Learning

Uczenie przez nagrody i kary.

| Plik                                  | Temat           | Kluczowe pojęcia                               |
| :------------------------------------ | :-------------- | :--------------------------------------------- |
| **44_RL_Q_Learning_FrozenLake.ipynb** | Q-Learning      | **Q-Table**, Równanie Bellmana, nagrody i kary |
| **45_RL_Deep_Q_Learning_DQN.ipynb**   | Deep Q-Learning | **DQN**, Replay Buffer, sieci neuronowe w RL   |

### 🧬 Algorytmy Zaawansowane

Specjalistyczne techniki i podejścia.

| Plik                                      | Temat                  | Kluczowe pojęcia                                            |
| :---------------------------------------- | :--------------------- | :---------------------------------------------------------- |
| **52_Genetic_Algorithms_Evolution.ipynb** | Algorytmy Genetyczne   | **Evolutionary Algorithms**, krzyżowanie, mutacja, selekcja |
| **53_Monte_Carlo_Simulation.ipynb**       | Symulacje Monte Carlo  | Symulacje probabilistyczne, analiza ryzyka                  |
| **54_FFT_Signal_Processing.ipynb**        | Przetwarzanie Sygnałów | **FFT**, Transformata Fouriera, analiza częstotliwości      |

### 💻 Inżynieria i Deployment

Praktyczne umiejętności produkcyjne.

| Plik                                         | Temat               | Kluczowe pojęcia                                               |
| :------------------------------------------- | :------------------ | :------------------------------------------------------------- |
| **25_Model_Persistence_Pickle_Joblib.ipynb** | Zapisywanie modeli  | **Pickle**, Joblib, serializacja obiektów                      |
| **30_Sklearn_Pipelines.ipynb**               | Rurociągi ML        | **Pipeline**, StandardScaler, data leakage prevention          |
| **31_Custom_Transformers.ipynb**             | Własne transformery | **BaseEstimator**, TransformerMixin, fit-transform pattern     |
| **28_Python_Dataclasses_for_ML.ipynb**       | Konfiguracja modeli | **Dataclasses**, structured configs, TrainingArguments pattern |
| **29_OOP_Classmethod_Staticmethod.ipynb**    | Wzorce projektowe   | **@classmethod**, @staticmethod, ModelLoader, factory pattern  |

---

## 🛠️ Technologie

Projekt oparty na standardowym stacku Data Science:

- **Python 3.x**
- **Pandas & NumPy** (Manipulacja danymi i obliczenia)
- **Scikit-Learn** (Algorytmy ML, Preprocessing, Metryki)
- **Matplotlib & Seaborn** (Wizualizacja danych)
- **SciPy** (Testy statystyczne)
- **PyTorch** (Deep Learning Framework)

## 🚀 Jak uruchomić?

1.  Sklonuj repozytorium:
    ```bash
    git clone https://github.com/takzen/ai-engineering-handbook
    ```
2.  Zainstaluj wymagane biblioteki:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn scipy torch
    ```
3.  Uruchom Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

---

Autor: Krzysztof Pika
