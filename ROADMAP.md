# 🗺️ AI Engineering Roadmap

Pełna ścieżka edukacyjna repozytorium AI-Engineering-Handbook. Każdy notatnik rozwiązuje konkretny problem inżynierski lub biznesowy.

---

## 00_KDE_Tutorial.ipynb – Rozkłady Gęstości

**Dlaczego?** Zwykły histogram jest "kanciasty" i zależy od tego, jak szerokie słupki ustawisz. Możesz przegapić ważne niuanse danych.

**Temat:** Kernel Density Estimation (KDE). Jak narysować gładką krzywą prawdopodobieństwa, która lepiej oddaje rzeczywistość niż słupki.

---

## 01_Correlations_and_Significance.ipynb – Zależności

**Dlaczego?** Musisz wiedzieć, które cechy wpływają na wynik, zanim zaczniesz trenować model.

**Temat:** Korelacja Pearsona (liniowa) vs Spearmana (rangowa). Heatmapy i pułapki interpretacji korelacji.

---

## 02_Advanced_Feature_Selection.ipynb – Selekcja Cech

**Dlaczego?** Więcej danych ≠ lepiej. Śmieciowe kolumny mylą model. Zwykła korelacja nie widzi zależności nieliniowych (np. paraboli).

**Temat:** SHAP, Mutual Information i Permutation Importance – nowoczesne metody oddzielania sygnału od szumu.

---

## 03_Encoding_Tutorial.ipynb – Zamiana Słów na Liczby

**Dlaczego?** Model matematyczny nie rozumie słowa "Czerwony" ani "BMW". Rozumie tylko liczby.

**Temat:** One-Hot Encoding, Label Encoding i Ordinal Encoding. Jak nie wprowadzić fałszywej hierarchii do danych.

---

## 04_Statistics_and_Scaling.ipynb – Skalowanie

**Dlaczego?** Algorytmy oparte na odległości (jak KNN) zwariują, jeśli jedna cecha ma zakres 0-1, a druga 0-10000.

**Temat:** Standaryzacja (StandardScaler) vs Normalizacja (MinMax). Kiedy używać którego?

---

## 05_Top_p_Top_k.ipynb – Generowanie Tekstu

**Dlaczego?** Dlaczego ChatGPT czasem jest kreatywny, a czasem precyzyjny?

**Temat:** Sampling. Sterowanie losowością wyboru kolejnego słowa (Top-k vs Nucleus Sampling).

---

## 06_Naive_Bayes_Spam.ipynb – Klasyfikacja Tekstu

**Dlaczego?** Czasami potrzebujesz super szybkiego i prostego modelu, który działa na małej ilości danych (np. prosty anty-spam).

**Temat:** Naiwny Klasyfikator Bayesa. Wykorzystanie prawdopodobieństwa warunkowego do oceny tekstu (Bag of Words).

---

## 07_Confusion_Matrix_Precision_Recall.ipynb – Metryki Błędów

**Dlaczego?** Dokładność (Accuracy) 99% to kłamstwo, jeśli w danych masz tylko 1% oszustw (model zgaduje "brak oszustwa" i ma rację, ale jest bezużyteczny).

**Temat:** Macierz Pomyłek. Precision (Precyzja), Recall (Czułość) i F1-Score.

---

## 08_Overfitting_Underfitting.ipynb – Diagnostyka

**Dlaczego?** Twój model działa świetnie na treningu, a tragicznie na produkcji.

**Temat:** Bias-Variance Tradeoff. Wizualizacja, jak zbyt skomplikowany model "wkuwa na pamięć" szum zamiast uczyć się reguł.

---

## 09_K_Means_Clustering.ipynb – Grupowanie

**Dlaczego?** Masz bazę klientów, ale nie masz etykiet (nie wiesz, kto jest kim). Chcesz ich podzielić na segmenty.

**Temat:** K-Means. Algorytm nienadzorowany i Metoda Łokcia (Elbow Method) do wyznaczania optymalnej liczby grup.

---

## 10_Decision_Trees.ipynb – Drzewa Decyzyjne

**Dlaczego?** Czasami musisz wiedzieć dokładnie, DLACZEGO model podjął decyzję (wymóg prawny/biznesowy).

**Temat:** Drzewa Decyzyjne. Algorytm "White Box", który tworzy czytelne reguły (if-else).

---

## 11_Embeddings_Vector_Space.ipynb – Matematyka Słów

**Dlaczego?** Jak sprawić, żeby komputer rozumiał, że "Król" i "Królowa" są blisko siebie znaczeniowo?

**Temat:** Embeddings. Reprezentacja słów jako wektorów w przestrzeni wielowymiarowej.

---

## 12_LLM_Temperature.ipynb – Parametry Modelu

**Dlaczego?** Jak działa suwak "Temperature" w API OpenAI?

**Temat:** Softmax z temperaturą. Jak matematycznie spłaszczyć lub wyostrzyć rozkład prawdopodobieństwa.

---

## 13_Missing_Data_Imputation.ipynb – Braki Danych

**Dlaczego?** Dane w świecie rzeczywistym są dziurawe. Usunięcie wierszy z brakami (NaN) to utrata cennych informacji.

**Temat:** Strategie Imputacji. Wypełnianie średnią, medianą oraz (najlepsze) inteligentne wypełnianie grupowe (Pandas transform).

---

## 14_Random_Forest_Ensemble.ipynb – Bagging

**Dlaczego?** Pojedyncze drzewo decyzyjne jest niestabilne i łatwo wkuwa dane na pamięć.

**Temat:** Las Losowy. Metoda "Siły Tłumu" – 100 drzew głosuje nad wynikiem, co wygładza błędy i daje stabilność.

---

## 15_Gradient_Descent.ipynb – Silnik Uczenia

**Dlaczego?** Jak właściwie model "wie", w którą stronę zmienić wagi, żeby zmniejszyć błąd?

**Temat:** Symulacja "schodzenia z góry". Zrozumienie Learning Rate i mechanizmu optymalizacji.

---

## 16_Neural_Network_Perceptron.ipynb – Pierwszy Neuron

**Dlaczego?** Żeby zrozumieć sieć, musisz zbudować jej najmniejszą cegiełkę.

**Temat:** Perceptron. Implementacja od zera i dowód, dlaczego pojedynczy neuron nie rozwiąże problemu XOR.

---

## 17_PCA_Dimensionality_Reduction.ipynb – Redukcja Wymiarów

**Dlaczego?** Masz 100 kolumn, a chcesz narysować wykres 2D.

**Temat:** PCA (Principal Component Analysis). Matematyczne "rzutowanie cienia" danych wielowymiarowych na płaszczyznę.

---

## 18_Cosine_Similarity_Search.ipynb – Wyszukiwanie

**Dlaczego?** Wyszukiwanie po słowach kluczowych jest słabe. Chcemy szukać po znaczeniu.

**Temat:** Podobieństwo Kosinusowe. Mierzenie kąta między wektorami zamiast odległości (kluczowe w RAG).

---

## 19_Cross_Validation.ipynb – Walidacja

**Dlaczego?** Jeden podział na Train/Test to hazard. Może miałeś szczęście przy losowaniu?

**Temat:** K-Fold Cross Validation. Trenowanie modelu 5 razy na różnych kawałkach danych, aby mieć pewność co do wyniku.

---

## 20_RAG_Architecture_Simulation.ipynb – RAG

**Dlaczego?** ChatGPT nie zna Twoich prywatnych dokumentów i halucynuje.

**Temat:** Retrieval Augmented Generation. Wyszukiwanie fragmentów wiedzy w bazie i doklejanie ich do promptu.

---

## 21_MLP_Neural_Network_XOR.ipynb – Sieci Wielowarstwowe

**Dlaczego?** Jak naprawić problem XOR, którego nie umiał rozwiązać Perceptron?

**Temat:** MLP (Multi-Layer Perceptron). Dodanie warstw ukrytych, które "wyginają przestrzeń".

---

## 22_Activation_Functions.ipynb – Funkcje Aktywacji

**Dlaczego?** Bez aktywacji sieć neuronowa to tylko mnożenie macierzy (funkcja liniowa).

**Temat:** Przegląd funkcji: ReLU (standard), Sigmoid (prawdopodobieństwo), Softmax (klasyfikacja).

---

## 23_Tokenization_GPT.ipynb – Tokenizacja

**Dlaczego?** Dlaczego modele AI nie umieją liczyć liter w słowach?

**Temat:** BPE (Byte Pair Encoding). Jak tekst jest szatkowany na tokeny przed wejściem do modelu (biblioteka Tiktoken).

---

## 24_Self_Attention_Mechanism.ipynb – Mechanizm Uwagi

**Dlaczego?** Jak model rozumie kontekst całego zdania naraz?

**Temat:** Matematyka Attention od zera. Macierze Query, Key, Value i iloczyn skalarny uwzględniający ważność słów.

---

## 25_Model_Persistence_Pickle_Joblib.ipynb – Zapisywanie Modeli

**Dlaczego?** Nie możesz trenować modelu od nowa za każdym razem, gdy klient wchodzi na stronę.

**Temat:** Serializacja. Zapisywanie wytrenowanego obiektu do pliku (.pkl, .joblib) i wczytywanie go na produkcji.

---

## 26_RAG_Chunking_Strategies.ipynb – Przygotowanie RAG

**Dlaczego?** Nie możesz wrzucić całej książki do bazy wektorowej w jednym kawałku.

**Temat:** Chunking. Strategie cięcia tekstu (Fixed Size, Recursive, Overlap), żeby nie gubić wątku.

---

## 27_Hyperparameter_Tuning_GridSearch.ipynb – Strojenie

**Dlaczego?** Zgadywanie, czy lepsze jest 10 drzew czy 50, to strata czasu.

**Temat:** Grid Search. Metoda "Brute Force" do automatycznego sprawdzania wszystkich kombinacji parametrów.

---

## 28_Python_Dataclasses_for_ML.ipynb – Czysty Kod (Config)

**Dlaczego?** Trzymanie parametrów modelu w zwykłym słowniku prowadzi do literówek i błędów, których nie widać od razu.

**Temat:** dataclasses. Typowanie silne w konfiguracji treningu, aby kod był bezpieczny i podpowiadał składnię.

---

## 29_OOP_Classmethod_Staticmethod.ipynb – Wzorce Projektowe

**Dlaczego?** Jak elegancko stworzyć model z pliku konfiguracyjnego, a jak ręcznie?

**Temat:** Metody fabryczne (@classmethod) i narzędziowe (@staticmethod) w kontekście budowania klas ML.

---

## 30_Sklearn_Pipelines.ipynb – Automatyzacja

**Dlaczego?** Jeśli robisz czyszczenie danych ręcznie przed modelem, na produkcji zapomnisz o jednym kroku i system padnie.

**Temat:** Budowa rurociągu (Pipeline), który skleja Imputer, Scaler i Model w jeden obiekt. Ochrona przed wyciekiem danych.

---

## 31_Custom_Transformers.ipynb – Własne Klasy

**Dlaczego?** Gotowe biblioteki nie mają funkcji "Wyczyść symbol waluty i usuń nawiasy".

**Temat:** Pisanie własnych klas dziedziczących po BaseEstimator, które można wpiąć w Pipeline Scikit-Learn.

---

## 32_PyTorch_Tensors_Autograd.ipynb – Silnik PyTorch

**Dlaczego?** Nie da się ręcznie liczyć pochodnych dla miliona wag w sieci neuronowej.

**Temat:** Tensors (macierze na GPU) i Autograd – mechanizm, który automatycznie śledzi obliczenia i liczy gradienty wstecz.

---

## 33_PyTorch_Neural_Network_Class.ipynb – Architektura

**Dlaczego?** Gotowe funkcje typu model.fit() to czarna skrzynka. Aby budować nowe rzeczy, musisz mieć kontrolę.

**Temat:** Budowa klasy nn.Module i ręczne pisanie pętli treningowej (Forward -> Loss -> Backward -> Step).

---

## 34_Regularization_Lasso_Ridge.ipynb – Regularyzacja

**Dlaczego?** Model, który uczy się za mocno (wielkie wagi), nie radzi sobie z nowymi danymi.

**Temat:** Lasso (L1) i Ridge (L2). Matematyczne "kary" nakładane na model, które zmuszają go do upraszczania rzeczywistości (i zerowania zbędnych cech w Lasso).

---

## 35_ANOVA_Hypothesis_Testing.ipynb – Testy A/B/C

**Dlaczego?** Porównywanie 3 grup (Lek A, Lek B, Placebo) za pomocą zwykłego testu parami to błąd statystyczny.

**Temat:** Analiza Wariancji (ANOVA), Test F oraz Test Tukeya (Post-hoc) do bezpiecznego porównywania wielu grup.

---

## 36_Market_Basket_Apriori.ipynb – Reguły Asocjacyjne

**Dlaczego?** Chcesz wiedzieć: "Kto kupił piwo, kupił też chipsy". To nie jest predykcja, to szukanie wzorców.

**Temat:** Algorytm Apriori. Zrozumienie metryk Support, Confidence i najważniejszego: Lift (siła reguły).

---

## 37_Gradient_Boosting_XGBoost.ipynb – Boosting

**Dlaczego?** Na danych tabelarycznych (Excel) sieci neuronowe często przegrywają. Królem jest Boosting.

**Temat:** XGBoost. Algorytm, w którym każde kolejne drzewo naprawia błędy poprzednika (sekwencyjne uczenie).

---

## 38_CNN_Computer_Vision.ipynb – Widzenie Komputerowe

**Dlaczego?** Zwykła sieć niszczy strukturę zdjęcia (spłaszcza je). Musimy widzieć kształty i krawędzie.

**Temat:** Sieci Splotowe (CNN). Warstwy Conv2d (filtry) i MaxPool (zmniejszanie).

---

## 39_RNN_LSTM_Sequence_Models.ipynb – Szeregi Czasowe

**Dlaczego?** Zwykła sieć nie pamięta, co było na poprzednim zdjęciu/kroku.

**Temat:** LSTM (Long Short-Term Memory). Sieć z "pamięcią", idealna do przewidywania giełdy, pogody czy tekstu.

---

## 40_Autoencoder_Anomaly_Detection.ipynb – Detekcja Anomalii

**Dlaczego?** Jak wykryć awarię silnika, skoro masz dane tylko z poprawnej pracy?

**Temat:** Autoenkoder. Uczenie nienadzorowane – sieć uczy się kompresować "normę". Jeśli nie potrafi czegoś skompresować (duży błąd), to znaczy, że to anomalia.

---

## 41_GAN_Generative_Adversarial_Network.ipynb – Generowanie Obrazu

**Dlaczego?** Jak zmusić sieć do tworzenia nowych rzeczy?

**Temat:** GAN. Wojna dwóch sieci: Fałszerza (Generator) i Policjanta (Dyskryminator).

---

## 42_Statistics_Masterclass.ipynb – Kompendium Statystyki

**Dlaczego?** Musisz wiedzieć, czy Twój wynik to "odkrycie", czy przypadek. Średnia arytmetyczna często kłamie przy zarobkach (Bill Gates w barze).

**Temat:** Rozkłady (Normalny), Prawo Wielkich Liczb, Test Shapiro-Wilka, Pułapki P-value i Paradoks Simpsona.

---

## 43_VAE_Variational_Autoencoder.ipynb – Latent Space

**Dlaczego?** Zwykła kompresja jest "sztywna". Nie da się płynnie zmienić cyfry 1 w 7.

**Temat:** VAE i Reparameterization Trick. Uczenie się rozkładu prawdopodobieństwa danych, co pozwala na "morfing".

---

## 44_RL_Q_Learning_FrozenLake.ipynb – Tabular RL

**Dlaczego?** Jak nauczyć robota chodzić, nie pokazując mu przykładów, tylko dając kary i nagrody?

**Temat:** Q-Learning. Tworzenie "ściągi" (Tabeli Q), która mówi, jaki ruch jest najlepszy w danej sytuacji.

---

## 45_RL_Deep_Q_Learning_DQN.ipynb – Deep RL

**Dlaczego?** W grze takiej jak StarCraft jest za dużo stanów, żeby zapisać je w tabeli.

**Temat:** DQN. Zastąpienie tabeli siecią neuronową, która "zgaduje" najlepszy ruch. Replay Buffer i Target Network dla stabilności.

---

## 46_Transformer_Block_From_Scratch.ipynb – Architektura GPT

**Dlaczego?** Attention to za mało. Prawdziwy Transformer to kanapka warstw.

**Temat:** Implementacja pełnego bloku: Attention -> LayerNorm -> FeedForward -> Residual Connection (Add).

---

## 47_SVM_Kernel_Trick.ipynb – SVM i Nieliniowość

**Dlaczego?** Jak rozdzielić dane (czerwone w środku, niebieskie na zewnątrz), których nie da się przeciąć prostą kreską?

**Temat:** Kernel Trick (RBF). Rzutowanie danych w wyższy wymiar, gdzie stają się separowalne liniowo.

---

## 48_tSNE_vs_PCA_Dimensionality_Reduction.ipynb – Wizualizacja Danych

**Dlaczego?** Masz dane 64-wymiarowe. Ekran jest 2D. PCA spłaszcza dane jak walec (gubiąc strukturę), t-SNE je "rozprostowuje".

**Temat:** Nieliniowa redukcja wymiarów. Porównanie, jak PCA i t-SNE radzą sobie z klastrowaniem cyfr (MNIST).

---

## 49_Object_Detection_IoU.ipynb – Detekcja Obiektów

**Dlaczego?** Sieć narysowała ramkę wokół kota. Skąd wiesz, czy trafiła dobrze?

**Temat:** Intersection over Union (IoU). Matematyka oceniania, jak bardzo dwie ramki na siebie nachodzą.

---

## 50_UNet_Image_Segmentation.ipynb – Segmentacja

**Dlaczego?** Czasem nie wystarczy wiedzieć "tu jest rak". Musisz wiedzieć dokładnie, który piksel to rak.

**Temat:** Architektura U-Net. Połączenia skrótowe (Skip Connections), które pozwalają sieci widzieć jednocześnie kontekst i precyzyjne detale.

---

## 51_Recommender_Systems_SVD.ipynb – Systemy Rekomendacyjne

**Dlaczego?** Masz miliony filmów i użytkowników. Tabela jest pusta w 99%. Jak zgadnąć ocenę filmu, którego nie widziałeś?

**Temat:** Faktoryzacja Macierzy (SVD). Rozbicie tabeli na ukryte cechy użytkowników i filmów.

---

## 52_Genetic_Algorithms_Evolution.ipynb – Algorytmy Genetyczne

**Dlaczego?** Gradient nie działa, gdy problem jest poszarpany lub dyskretny (np. co spakować do plecaka).

**Temat:** Ewolucja. Symulacja populacji, krzyżowania i mutacji w celu znalezienia optymalnego rozwiązania bez użycia pochodnych.

---

## 53_Monte_Carlo_Simulation.ipynb – Symulacje Ryzyka

**Dlaczego?** W finansach nie pytamy "ile zarobię?", tylko "jaka jest szansa, że zbankrutuję?". Przeszłość nie gwarantuje przyszłości.

**Temat:** Generowanie 1000 alternatywnych scenariuszy giełdowych (Geometryczne Ruchy Browna) i obliczanie VaR (Value at Risk).

---

## 54_FFT_Signal_Processing.ipynb – Przetwarzanie Sygnału

**Dlaczego?** Dane to nie tylko tabelki, to też dźwięk i wibracje. Na wykresie czasowym szumu nie widać.

**Temat:** Szybka Transformata Fouriera (FFT). Zamiana osi czasu na oś częstotliwości, aby "zobaczyć" i wyciąć pisk z nagrania.

---

## 55_LoRA_Fine_Tuning_Math.ipynb – Fine-Tuning

**Dlaczego?** Douczanie modelu GPT ważącego 100GB jest niemożliwe na laptopie.

**Temat:** LoRA (Low-Rank Adaptation). Matematyczny trik polegający na douczaniu tylko malutkich macierzy-nakładek (Adapterów).

---

## 56_Positional_Encoding_Transformer.ipynb – Czas w Transformerze

**Dlaczego?** Transformer czyta całe zdanie naraz (równolegle). Nie wie, co było wcześniej, a co później.

**Temat:** Positional Encoding. Dodawanie fal sinusoidalnych do wektorów słów, aby nadać im "sygnaturę czasu".

---

## 57_Metric_Learning_Siamese.ipynb – FaceID

**Dlaczego?** Telefon nie ma w bazie milionów twarzy. On sprawdza, czy Twoja twarz jest podobna do tej zapisanej.

**Temat:** Sieci Syjamskie i Triplet Loss. Uczenie sieci mierzenia odległości między obiektami, a nie ich klasyfikacji.

---

## 58_Graph_Neural_Networks_GNN.ipynb – Grafy

**Dlaczego?** Social media i chemia to nie tabelki. To relacje (kto zna kogo, jaki atom wiąże się z jakim).

**Temat:** Message Passing. Jak węzły w grafie wymieniają się informacjami ze swoimi sąsiadami.

---

## 59_Model_Quantization_INT8.ipynb – Kompresja Modeli

**Dlaczego?** Wielkie modele nie mieszczą się w pamięci telefonu.

**Temat:** Matematyka rzutowania liczb zmiennoprzecinkowych (FP32) na całkowite (INT8). Obliczanie Scale i Zero Point.

---

## 60_Bayesian_Optimization_Optuna.ipynb – Optymalizacja

**Dlaczego?** Grid Search sprawdza wszystko "na siłę" (strata prądu). Random Search to hazard.

**Temat:** Optuna. Algorytm, który uczy się na błędach i inteligentnie dobiera parametry modelu, żeby zmaksymalizować wynik.

---

## 61_Normalization_Layers_BN_vs_LN.ipynb – Stabilizacja

**Dlaczego?** Bez normalizacji głębokie sieci przestają się uczyć (wybuchające gradienty).

**Temat:** Różnica między BatchNorm (dla obrazów), LayerNorm (dla tekstu/Transformerów) i InstanceNorm.

---

## 62_Time_Series_Decomposition_STL.ipynb – Szeregi Czasowe

**Dlaczego?** Szef pyta: "Dlaczego sprzedaż spadła?". Musisz wiedzieć, czy to trend (kryzys), czy sezonowość (koniec świąt).

**Temat:** Dekompozycja STL. Rozbicie wykresu na trzy składniki: Trend, Sezonowość i Reszty (Szum/Anomalie).

---

## 63_Diffusion_Models_DDPM.ipynb – Matematyka Stable Diffusion

**Dlaczego?** GANy (notatnik 41) są trudne w treningu. Dzisiaj króluje Dyfuzja.

**Temat:** Zaimplementujemy od zera proces Forward Diffusion (dodawanie szumu do zdjęcia aż zniknie) i Reverse Diffusion (nauczenie sieci, jak "odszumiać" biały szum, żeby powstał obraz).

---

## 64_Knowledge_Distillation.ipynb – Nauczyciel i Uczeń

**Dlaczego?** Kwantyzacja (notatnik 59) to nie jedyny sposób na małe modele. Możemy wziąć wielkiego modelu (Teacher) i kazać mu uczyć małego modelu (Student).

**Temat:** Soft Labels i Temperature. Student nie uczy się tylko "To jest kot", ale uczy się od Nauczyciela, że "To jest kot, ale trochę przypomina psa".

---

## 65_HNSW_Vector_Database.ipynb – Silnik Baz Wektorowych

**Dlaczego?** W RAG (notatnik 20) liczyliśmy cosinusy dla wszystkich par. To działa dla 10k plików. Dla 100 milionów potrzebujesz HNSW.

**Temat:** Zbudujemy (uproszczoną) strukturę grafową, która pozwala znaleźć najbliższy wektor w czasie logarytmicznym, skacząc po "autostradach" między węzłami.

---

## 66_Kalman_Filter.ipynb – Śledzenie w czasie rzeczywistym

**Dlaczego?** GPS w Twoim telefonie kłamie (ma błąd 5m). Jakim cudem nawigacja pokazuje Cię idealnie na drodze?

**Temat:** Algorytm, który łączy Predykcję (gdzie powinieneś być wg fizyki) z Pomiarem (gdzie jesteś wg czujnika) i magicznie usuwa szum. Fundament robotyki.

---

## 67_Contrastive_Learning_SimCLR.ipynb – SimCLR / CLIP

**Dlaczego?** Jak nauczyć sieć rozumieć obrazy bez etykiet?

**Temat:** Uczenie kontrastowe. Bierzemy zdjęcie psa, robimy mu 2 różne filtry (obrót, kolor). Mówimy sieci: "Te dwa różne obrazy to to samo. Zbliż ich wektory do siebie".

---

## 68_PPO_RLHF.ipynb – Jak powstał ChatGPT?

**Dlaczego?** Zwykły model językowy umie tylko dokończyć zdanie. Żeby był asystentem, musi przejść RLHF (Reinforcement Learning from Human Feedback).

**Temat:** Zrozumienie mechanizmu PPO – jak nagradzać model językowy za dobre odpowiedzi, nie psując jego zdolności językowych (Kullback-Leibler penalty).

---

## 69_Data_Drift_Detection.ipynb – KS-Test & PSI

**Dlaczego?** Wytrenowałeś model na danych z 2023. Jest 2025. Model działa źle. Dlaczego? Bo dane się zmieniły (Drift).

**Temat:** Matematyczne wykrywanie, czy "świat się zmienił". Test Kolmogorova-Smirnova i Population Stability Index. Obowiązkowe w bankowości.

---

## 70_Vision_Transformers_ViT.ipynb – Koniec ery CNN?

**Dlaczego?** CNN (sploty) patrzą lokalnie (okno 3x3). Transformery patrzą globalnie. Od 2021 roku ViT wypiera CNN w rozpoznawaniu obrazów.

**Temat:** Jak pociąć zdjęcie na "słowa" (Patches), dodać im pozycję i wrzucić do Transformera, którego zbudowaliśmy w notatniku 46.

---

## 71_Flash_Attention_KV_Cache.ipynb – Jak działają szybkie LLM?

**Dlaczego?** Zwykłe Attention ma złożoność O(N²). Dla długich tekstów pamięć GPU wybucha.

**Temat:** Zrozumienie optymalizacji sprzętowej (Tiling) i mechanizmu Key-Value Cache (nie licz tego samego dwa razy), który pozwala ChatGPT odpisywać w czasie rzeczywistym.

---

## 72_Mamba_SSM.ipynb – State Space Models

**Dlaczego?** Transformer zatyka się przy długich tekstach. Mamba ma złożoność liniową i pamięta nieskończenie długo.

**Temat:** Architektura SSM. Dyskretyzacja równań różniczkowych w sieci neuronowej.

---

## 73_Mixture_of_Experts_MoE.ipynb – Architektura GPT-4

**Dlaczego?** GPT-4 to nie jeden wielki model, ale "rada ekspertów".

**Temat:** Budowa warstwy Gating Network (Routera), która decyduje, który mały model (Ekspert) ma obsłużyć dane zapytanie.

---

## 74_Liquid_Neural_Networks.ipynb – LFC

**Dlaczego?** Sieci neuronowe są statyczne po treningu. Liquid Networks zmieniają swoje wagi w czasie rzeczywistym, adaptując się do otoczenia.

**Temat:** Implementacja sieci opartej na równaniach różniczkowych, która działa świetnie w robotyce i dronach.

---

## 75_Meta_Learning_MAML.ipynb – Model-Agnostic Meta-Learning

**Dlaczego?** Człowiek uczy się po jednym przykładzie. Sieć potrzebuje tysięcy.

**Temat:** Model-Agnostic Meta-Learning. Trenowanie sieci tak, aby jej wagi były "łatwe do szybkiego douczenia" na nowym zadaniu w 5 krokach.

---

## 76_LangChain_ReAct.ipynb – Reason + Act

**Dlaczego?** Model językowy nie ma dostępu do kalkulatora ani Google.

**Temat:** Budowa pętli Agenta: Myśl -> Wybierz Narzędzie -> Wykonaj Akcję -> Obserwuj Wynik -> Odpowiedz.

---

## 77_Prompt_Engineering_CoT.ipynb – Chain of Thought

**Dlaczego?** Proste pytanie "Ile to 23\*45?" często myli model.

**Temat:** Implementacja CoT i Tree of Thoughts. Wymuszanie na modelu "myślenia na głos" przed podaniem odpowiedzi.

---

## 78_RAG_Evaluation_RAGAS.ipynb – RAGAS

**Dlaczego?** Zbudowałeś RAG (notatnik 20). Skąd wiesz, czy jest dobry?

**Temat:** Metryki: Faithfulness (czy odpowiedź wynika z kontekstu?) i Answer Relevance (czy odpowiada na pytanie?).

---

## 79_Product_Quantization_IVF_PQ.ipynb – Vector Database

**Dlaczego?** HNSW (notatnik 65) jest szybkie, ale żre RAM.

**Temat:** Kompresja wektorów (Product Quantization). Jak zmieścić miliard wektorów w pamięci, tracąc minimalnie na precyzji (technologia FAISS).

---

## 80_Speculative_Decoding.ipynb – Speculative Decoding

**Dlaczego?** Wielki model jest wolny. Mały model jest szybki, ale głupi.

**Temat:** Użycie małego modelu do "zgadywania" 5 kolejnych słów, a dużego tylko do ich "zatwierdzania". Przyspieszenie generowania tekstu 2-3x.

---

## 81_Audio_Mel_Spectrograms.ipynb – Spektrogramy Melowe

**Dlaczego?** Surowe audio to chaos. Modele (Whisper) "patrzą" na dźwięk.

**Temat:** Zamiana audio na Mel-Spectrogram (skala logarytmiczna, dopasowana do ucha ludzkiego).

---

## 82_ASR_CTC_Loss.ipynb – Connectionist Temporal Classification

**Dlaczego?** Plik audio trwa 5 sekund, tekst ma 10 liter. Jak to zmapować bez wyrównywania?

**Temat:** Matematyka CTC. Algorytm, który pozwala sieci "pluć" literami w odpowiednich momentach i usuwać duplikaty ("heeeelllo" -> "hello").

---

## 83_Text_to_Speech_Tacotron.ipynb – Tacotron Concept

**Dlaczego?** Chcesz, żeby AI do Ciebie mówiło.

**Temat:** Architektura Encoder-Decoder z mechanizmem Attention, która zamienia litery na spektrogramy audio.

---

## 84_CLIP_Multimodal.ipynb – Contrastive Language-Image Pretraining

**Dlaczego?** Jak połączyć tekst z obrazem? (Podstawa DALL-E).

**Temat:** Trening modelu, który umie powiedzieć, jak bardzo zdjęcie psa pasuje do podpisu "Pies na plaży".

---

## 85_NeRF_Neural_Radiance_Fields.ipynb – Neural Radiance Fields

**Dlaczego?** Jak z 5 zdjęć 2D zrobić model 3D?

**Temat:** Reprezentacja sceny jako funkcji F(x,y,z)→(Kolor,Gęstość). Ray Marching (strzelanie promieniami przez piksele).

---

## 86_PointNet_Point_Clouds.ipynb – Chmury Punktów

**Dlaczego?** Lidary w samochodach autonomicznych widzą chmury punktów, a nie zdjęcia.

**Temat:** Architektura, która przetwarza zbiór punktów 3D (niezależnie od ich kolejności) i klasyfikuje obiekt (np. "To jest krzesło").

---

## 87_Graph_Attention_Networks_GAT.ipynb – Graph Attention Networks

**Dlaczego?** Zwykły GCN (notatnik 58) traktuje wszystkich sąsiadów tak samo.

**Temat:** Dodanie mechanizmu Attention do grafów. Węzeł sam decyduje, który sąsiad jest ważny.

---

## 88_Temporal_Fusion_Transformer_TFT.ipynb – TFT

**Dlaczego?** LSTM jest stary. TFT to obecny SOTA w prognozowaniu (np. zapotrzebowania na prąd).

**Temat:** Transformer, który obsługuje wiele horyzontów czasowych i zmienne statyczne (np. lokalizacja sklepu) oraz dynamiczne (pogoda).

---

## 89_NeuralProphet.ipynb – NeuralProphet

**Dlaczego?** Prophet od Facebooka jest super, ale wolny.

**Temat:** Hybryda klasycznej dekompozycji (Trend/Sezon) z sieciami neuronowymi w PyTorch.

---

## 90_Isolation_Forest.ipynb – Isolation Forest

**Dlaczego?** Autoenkoder (notatnik 40) jest super, ale czasem potrzebujesz szybkiego algorytmu drzewiastego do anomalii.

**Temat:** Algorytm, który próbuje "odizolować" punkt losowymi cięciami. Im szybciej się to uda, tym bardziej punkt jest podejrzany (anomalia).

---

## 91_LIME.ipynb – Local Interpretable Model-agnostic Explanations

**Dlaczego?** SHAP jest dokładny, ale wolny.

**Temat:** Jak wyjaśnić decyzję skomplikowanego modelu, trenując wokół jednej próbki mały, prosty model liniowy.

---

## 92_Counterfactual_Explanations.ipynb – Counterfactual Explanations

**Dlaczego?** Klient pyta: "Co muszę zmienić, żeby dostać kredyt?".

**Temat:** Generowanie alternatywnych scenariuszy: "Gdybyś zarabiał 500 zł więcej, dostałbyś kredyt".

---

## 93_Fairness_Bias_Detection.ipynb – Fairness & Bias Detection

**Dlaczego?** Twój model może dyskryminować.

**Temat:** Metryki sprawiedliwości (Disparate Impact, Equal Opportunity). Sprawdzanie, czy model traktuje tak samo różne grupy społeczne.

---

## 94_Causal_Inference.ipynb – Do-Calculus

**Dlaczego?** Korelacja to nie przyczynowość. Czy reklama zwiększyła sprzedaż, czy po prostu były święta?

**Temat:** Drabina Pearla. Odróżnianie obserwacji P(Y∣X) od interwencji.

---

## 95_Model_Pruning.ipynb – Przycinanie

**Dlaczego?** Wiele wag w sieci jest bliskich zeru. Są niepotrzebne.

**Temat:** Zerowanie najsłabszych połączeń w sieci neuronowej i dotrenowanie reszty. Odchudzanie modelu o 50-90%.

---

## 96_ONNX_Runtime.ipynb – ONNX Runtime

**Dlaczego?** PyTorch jest świetny do treningu, ale ciężki na produkcji.

**Temat:** Eksport modelu do formatu ONNX (Open Neural Network Exchange), który działa na wszystkim (od serwera po przeglądarkę).

---

## 97_Federated_Learning.ipynb – Federated Learning

**Dlaczego?** Dane (np. medyczne) nie mogą opuścić telefonu pacjenta.

**Temat:** Trenowanie modelu na tysiącach urządzeń lokalnie i przesyłanie do serwera tylko aktualizacji wag (a nie danych).

---

## 98_Adversarial_Attacks_FGSM.ipynb – FGSM

**Dlaczego?** Jak oszukać sieć, zmieniając jeden piksel?

**Temat:** Fast Gradient Sign Method. Generowanie szumu, który jest niewidoczny dla oka, a sprawia, że AI widzi banana zamiast pistoletu.

---

## 99_Neural_Architecture_Search_NAS.ipynb – NAS

**Dlaczego?** Po co projektować sieć ręcznie, skoro AI może zaprojektować AI?

**Temat:** Prosty algorytm, który sam buduje architekturę sieci (dobiera liczbę warstw i filtrów) metodą ewolucyjną.

---

## 100_World_Models.ipynb – Model-Based RL

**Dlaczego?** Agent RL (DQN) uczy się wolno, bo musi "przeżyć" grę.

**Temat:** Agent, który najpierw uczy się "modelu świata" (jak działa fizyka), a potem "śni" (symuluje) w głowie przyszłość, żeby podjąć decyzję.

---

## 101_MLOps_Platform.ipynb – The Grand Finale: End-to-End MLOps Platform

**Dlaczego?** Zwieńczenie wszystkiego.

**Temat:** Projekt spinający wiedzę: Trening modelu -> Wersjonowanie (MLflow/DVC concept) -> Testy (PyTest) -> Serwowanie (FastAPI) -> Monitoring (Drift). (W formie kodu Python, symulacja całej platformy).
