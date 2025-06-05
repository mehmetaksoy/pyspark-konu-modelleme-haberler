# PySpark ile Haber Başlıkları Üzerinde Konu Modelleme (K-means Clustering)

Bu proje, "abcnews-date-text.csv" veri setindeki haber başlıklarını kullanarak Doğal Dil İşleme (NLP) teknikleri ve PySpark üzerinde K-means kümeleme algoritması ile konu modellemesi yapmayı amaçlamaktadır. Proje, metin verilerinin temizlenmesi, özellik çıkarımı (TF-IDF ve Word2Vec) ve kümeleme adımlarını içermektedir. Bu notebook, bir Databricks ortamında çalıştırılmak üzere hazırlanmıştır.

## 🎯 Projenin Amacı

Temel amaç, geniş bir haber başlığı koleksiyonunu anlamsal olarak benzer gruplara (konulara) ayırmaktır. Bu sayede, veri setindeki ana temalar ve konular hakkında fikir edinilebilir.

## 🛠️ Kullanılan Teknolojiler ve Yöntemler

* **Dil:** Python
* **Ana Kütüphaneler:**
    * `PySpark`: Büyük veri işleme ve dağıtık hesaplama için.
        * `pyspark.sql`: DataFrame işlemleri için.
        * `pyspark.ml.feature`: Metin özellik çıkarma (Tokenizer, StopWordsRemover, CountVectorizer, IDF, Word2Vec) için.
        * `pyspark.ml.clustering`: K-means kümeleme için.
        * `pyspark.ml.evaluation`: Kümeleme değerlendirme (ClusteringEvaluator) için.
    * `nltk`: Kelime köklerini bulma (SnowballStemmer) için.
    * `pandas`, `numpy`: Veri manipülasyonu ve sayısal işlemler için (özellikle elbow metodu sonuçlarını göstermek gibi yardımcı görevlerde).
    * `matplotlib`, `seaborn`: (Notebook'ta import edilmiş ancak aktif grafik çizimi görülmemiştir, potansiyel görselleştirmeler için dahil edilebilir).
* **Veri Kaynağı:** `abcnews-date-text.csv` (ABC News haber başlıkları).
* **NLP Ön İşleme Adımları:**
    1.  Metin Temizleme: Küçük harfe çevirme, boşlukları ve noktalama işaretlerini kaldırma.
    2.  Tokenizasyon: Metinleri kelimelere ayırma.
    3.  Stopwords (Etkisiz Kelimeler) Kaldırma.
    4.  Stemming (Kök Bulma): İngilizce kelimeler için SnowballStemmer.
* **Özellik Çıkarma (Vektörleştirme):**
    1.  **TF-IDF (Term Frequency-Inverse Document Frequency)**
    2.  **Word2Vec** (Alternatif bir yöntem olarak)
* **Kümeleme Algoritması:** K-means
* **Değerlendirme Metriği:** Silhouette Score
* **Optimal Küme Sayısı Belirleme:** Elbow Method (Word2Vec için uygulanmış)
* **Çalışma Ortamı:** Databricks ( `%sh`, `%fs` ve `dbutils` komutları kullanılmıştır).

## 📝 Veri Seti

Projede kullanılan veri seti, Avustralya ABC News tarafından yayınlanan haber başlıklarını içermektedir. Veri seti `publish_date` (yayın tarihi) ve `headline_text` (haber başlığı metni) olmak üzere iki sütundan oluşur. Veri seti, bir Google Drive linkinden indirilmektedir.

## 🌊 İş Akışı (Pipeline)

1.  **Veri Yükleme ve Hazırlık:**
    * Haber başlıkları veri seti `%sh curl` komutu ile Google Drive'dan Databricks sürücü düğümünün yerel `/tmp` klasörüne indirilir.
    * `dbutils.fs.mv` komutu ile dosya DBFS'e (`dbfs:/datasets/`) taşınır.
    * Veri, `spark.read.load` ile bir Spark DataFrame'e yüklenir.
    * Veri setinin yapısı, boyutu incelenir ve `headline_text` bazında yinelenen kayıtlar temizlenir. Eksik değer kontrolü yapılır.

2.  **Doğal Dil Ön İşleme (`clean_text` fonksiyonu):**
    * `headline_text` sütunundaki metinler küçük harfe çevrilir.
    * Baştaki/sondaki boşluklar ve noktalama işaretleri kaldırılır.
    * Metinler, PySpark `Tokenizer` ile kelimelere (token) ayrılır.
    * İngilizce için yaygın etkisiz kelimeler (stopwords) PySpark `StopWordsRemover` ile çıkarılır.
    * NLTK `SnowballStemmer` kullanılarak kelimeler köklerine indirgenir (PySpark UDF ile uygulanır).

3.  **Özellik Çıkarma (Feature Engineering):**
    * **TF-IDF Vektörleri (`extract_tfidf_features` fonksiyonu):**
        * Ön işlenmiş (köklerine ayrılmış) metinlerden, PySpark `CountVectorizer` ve `IDF` kullanılarak TF-IDF vektörleri oluşturulur.
        * Oluşan vektörlerden sıfır uzunluklu olanlar (yani hiçbir özellik içermeyenler) filtrelenir.
    * **Word2Vec Vektörleri (`extract_w2v_features` fonksiyonu):**
        * (Etkisiz kelimeleri çıkarılmış ancak köklerine ayrılmamış) metinlerden PySpark `Word2Vec` modeli eğitilir ve kelime/belge vektörleri elde edilir.
        * Eğitilen model ile kelime sinonimleri bulma örneği gösterilir.

4.  **K-means Kümeleme (`k_means` fonksiyonu):**
    * TF-IDF özellikleri kullanılarak PySpark `KMeans` ile model eğitilir.
    * Belirlenen sayıda (`N_CLUSTERS`) küme oluşturulur.

5.  **Kümeleme Değerlendirmesi:**
    * `evaluate_k_means` fonksiyonu ile Silhouette Score kullanılarak kümeleme kalitesi değerlendirilir.
    * Her kümedeki öğe sayısı ve her kümeden örnek haber başlıkları incelenir.

6.  **Optimal Küme Sayısı (Elbow Method - Word2Vec için):**
    * `elbow_method` fonksiyonu ile Word2Vec özellikleri üzerinde farklı K değerleri için K-means çalıştırılır ve Sum of Squared Errors (SSE) değerleri toplanarak dirsek grafiği için veri hazırlanır.

## 🚀 Kurulum ve Çalıştırma (Databricks Ortamında)

1.  **Notebook İçe Aktarma:** Bu `nlp_kmeans.ipynb` dosyasını Databricks çalışma alanınıza (workspace) import edin.
2.  **Kütüphane Kurulumu:**
    * Notebook'un ilk hücresindeki `pip install nltk` komutu, NLTK kütüphanesini küme (cluster) ortamına kuracaktır. Eğer küme yapılandırmasında kütüphane zaten kurulu değilse bu adım gereklidir.
    * Diğer kütüphaneler (`pyspark`, `pandas` vb.) genellikle Databricks ortamlarında standart olarak bulunur.
3.  **Küme (Cluster) Hazırlığı:** Notebook'u çalıştıracağınız Databricks kümesinin NLTK kütüphanesini ve bağımlılıklarını içerdiğinden emin olun.
4.  **Çalıştırma:**
    * Hücreleri sırayla çalıştırın.
    * Veri indirme ve DBFS'e taşıma işlemleri notebook içindeki `%sh` ve `dbutils` komutları ile otomatik olarak yapılır.
    * NLP ön işleme, özellik çıkarma ve kümeleme adımları Spark üzerinde dağıtık olarak çalışacaktır.

## 📊 Sonuçların Yorumlanması

* **Silhouette Score:** Elde edilen skorun (-1 ile 1 arasında) 1'e yakın olması daha iyi bir kümelemeyi, 0'a yakın olması kümelerin üst üste bindiğini, negatif değerler ise yanlış kümelemeyi işaret eder. TF-IDF ile elde edilen skor (~0.039) çok yüksek olmasa da, metin kümelemede bu tür skorlar görülebilir ve iyileştirme potansiyeli olduğunu gösterir.
* **Küme İçerikleri:** Her kümeden rastgele seçilen haber başlıkları incelenerek kümelerin anlamsal tutarlılığı ve hangi konuları temsil ettiği hakkında fikir edinilebilir.
* **Elbow Method Grafiği:** Word2Vec için çizilecek dirsek grafiği (bu notebook'ta grafik çizimi eklenmemiş ama verisi toplanmış), SSE'deki azalmanın yavaşladığı "dirsek" noktasını bularak optimal K sayısını tahmin etmeye yardımcı olur.

## 💡 Olası Geliştirmeler

* Farklı NLP ön işleme adımları denenebilir (örneğin, Lemmatization yerine Stemming veya tam tersi, farklı n-gram aralıkları).
* TF-IDF için `vocabSize`, `minDF` gibi parametreler optimize edilebilir.
* Word2Vec için `vectorSize`, `minCount`, `windowSize` gibi parametreler optimize edilebilir.
* K-means dışındaki kümeleme algoritmaları (örneğin, Latent Dirichlet Allocation - LDA, DBSCAN) denenebilir.
* Kümelerin daha iyi yorumlanabilmesi için her kümedeki en karakteristik kelimeler (örneğin, TF-IDF skorları en yüksek kelimeler) çıkarılabilir.
* Sonuçların görselleştirilmesi (örneğin, t-SNE veya UMAP ile boyut indirgeme sonrası küme grafikleri).