# 🚗 Privacy-Preserving Autonomous Driving con YOLOv9
**Un approccio "Data-Centric" per l'Object Detection su Lenti Fisheye**

Questo repository contiene il codice sorgente, i notebook sperimentali e i modelli addestrati (pesi `.pt`) del progetto **PP4AV-YOLOv9**. 
L'obiettivo del progetto è l'adattamento dell'architettura state-of-the-art **YOLOv9** all'anonimizzazione dei dati in domini ottici severi.

## 🎯 Il Contesto e il Problema
I moderni sistemi di guida autonoma raccolgono inevitabilmente dati sensibili come volti e targhe. Il GDPR rende l'anonimizzazione un vincolo legale stringente. 
Il vero problema ("corner cases") si presenta con l'uso di telecamere grandangolari o fisheye a 360 gradi, che introducono forti distorsioni radiali alterando le proporzioni ai bordi.
I modelli tradizionali tendono a fallire in queste situazioni critiche.

Questo progetto indaga se YOLOv9 possa eguagliare o superare le metriche di una complessa baseline YOLOX del seguente lavoro di ricerca:
```text
@article{PP4AV2022,
  title = {PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving},
  author = {Linh Trinh, Phuong Pham, Hoang Trinh, Nguyen Bach, Dung Nguyen, Giang Nguyen, Huy Nguyen},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year = {2023}
}
```
Il tutto utilizzando esclusivamente il dataset PP4AV, senza ricorrere a enormi moli di video esterni o tecniche di "Knowledge Distillation", ma affidandosi alla *Data Efficiency*.

## 📊 Il Dataset (PP4AV)
Il progetto si basa sul dataset **PP4AV** (Privacy-Preserving for Autonomous Vehicles).
* È composto da **3.447 immagini** annotate, campionate da video di guida in sei città europee.
* Include scenari complessi come la guida notturna (Netherlands_night) e immagini con distorsione ottica (Fisheye).
* Presenta un forte sbilanciamento: le Targhe rappresentano il 67.3% delle annotazioni, i Volti il 32.7%.
* Si tratta di un task di *Small Object Detection* estremo (l'area media di una targa è dello 0.0634% rispetto all'immagine).

## 📂 Struttura del Repository

```text
📦 PP4AV-YOLOv9
 ┣ 📜 Documentazione.pdf            # File di report e documentazione teorica
 ┣ 📂 EDA                           # Exploratory Data Analysis
 ┃ ┗ 📜 pp4av-eda.ipynb             # Analisi statistica e visiva del dataset
 ┣ 📂 Primo_ciclo_sperimentale      # Generalizzazione "Zero-Shot"
 ┃ ┣ 📜 pp4av-split1.ipynb          # Partizionamento dati (Fisheye isolato)
 ┃ ┣ 📜 pp4av-yolov9-finetuning-split1.ipynb 
 ┃ ┣ 📜 pp4av-test-yolov9-split1.ipynb       
 ┃ ┣ 📂 analisi fine-tuning         # Curve di Loss, PR Curve, F1 Curve e matrici di confusione normalizzate del training
 ┃ ┣ 📂 analisi test                # Curve di Loss, PR Curve, F1 Curve e matrici di confusione normalizzate del test
 ┃ ┗ 📜 best.pt                     # Pesi del modello ottimizzato
 ┣ 📂 Secondo_ciclo_sperimentale    # Addestramento "Mixed-Domain"
 ┃ ┣ 📜 pp4av-split2.ipynb          # Partizionamento dati uguale su tutti i domini
 ┃ ┣ 📜 pp4av-yolov9-finetuning-split2.ipynb 
 ┃ ┣ 📜 pp4av-test-yolov9-split2.ipynb
 ┃ ┣ 📂 analisi fine-tuning
 ┃ ┣ 📂 analisi test
 ┃ ┗ 📜 best.pt
 ┗ 📂 Approccio_Data-Centric        # Generazione Sintetica e Bilanciamento
   ┣ 📜 pp4av-datacentric-split3.ipynb       # Motore di distorsione a barile (Albumentations)
   ┣ 📜 pp4av-finetuning-datacentric-split3.ipynb 
   ┣ 📜 pp4av-test-yolov9-split3.ipynb
   ┗ 📜 best.pt                     # Pesi del modello finale (Definitivo)
```

## 🔬 Metodologia: I Tre Cicli Sperimentali
Il flusso di lavoro si articola in tre fasi:

1. **Ciclo 1 (Valutazione Zero-Shot):** addestramento esclusivo su immagini a prospettiva lineare. Il test sul set Fisheye isolato ha mostrato un crollo verticale delle prestazioni causato dall'"Overfitting Geometrico".
2. **Ciclo 2 (Mixed-Domain):** addestramento su un set misto (lineare e grandangolare). Ha curato l'overfitting geometrico per le targhe (AP 83.31%), ma la classe "Volto" ha registrato un calo rispetto alla baseline (AP 43.68%) a causa della difficoltà geometrica combinata allo sbilanciamento.
3. **Ciclo 3 (Data-Centric):** introduzione della generazione sintetica tramite la libreria Albumentations per applicare distorsioni a barile (OpticalDistortion). A questo si aggiunge un bilanciamento della funzione di Loss (cls = 1.5) per forzare il modello a concentrarsi sulla classe minoritaria.


## ⚙️ Guida passo-passo alla Riproducibilità

Questo progetto è stato sviluppato nativamente in ambiente **Kaggle** per sfruttare l'accelerazione GPU (NVIDIA Tesla T4). Di seguito i passaggi per eseguire autonomamente il codice.

### 1. Download del Dataset Originale
Prima di avviare i notebook, è necessario ottenere i dati grezzi:
1. Utilizzare il dataset ufficiale **PP4AV**.
2. Scarica il dataset e posizionalo nel tuo ambiente (o collegalo direttamente al tuo notebook se stai lavorando su Kaggle).

### 2. Impostazione dei percorsi (Path)
Ogni notebook inizia con la definizione delle variabili di percorso. Se esegui il codice in locale o su un ambiente diverso da Kaggle, assicurati di modificare queste variabili.
Cambiare anche i percorsi di output presenti nel codice.

### 3. Esecuzione del Flusso di Lavoro (Esempio per il Ciclo 3)
Per riprodurre il modello finale (Approccio Data-Centric), segui questo ordine rigoroso all'interno della cartella `Approccio_Data-Centric`:

1. **Generazione e Split Dati:** apri ed esegui `pp4av-datacentric-split3.ipynb`. Questo script applicherà le distorsioni di Albumentations, bilancerà le classi e genererà una nuova cartella con la struttura pronta per YOLO e i file `.yaml` di configurazione.
2. **Addestramento (Fine-Tuning):** apri `pp4av-finetuning-datacentric-split3.ipynb`. Assicurati di avere una GPU attiva. Esegui il notebook per scaricare i pesi pre-addestrati di YOLOv9c e avviare l'addestramento di 50 epoche con la *Loss* bilanciata. I pesi finali verranno salvati in una cartella simile a `runs/detect/train/weights/best.pt`.
3. **Test e Inferenza:** apri `pp4av-test-yolov9-split3.ipynb`. Assicurati che il percorso dei pesi punti al file `best.pt` (fornito nel repository o appena generato da te). Esegui il notebook per calcolare programmaticamente AP, AR e generare le matrici di confusione sui domini Normal e Fisheye.

### 4. Utilizzo Immediato (Inference)
Se desideri saltare la fase di addestramento e testare subito le capacità del modello su tue immagini, puoi caricare il file `best.pt` fornito nella cartella del Ciclo 3 e passarlo direttamente al modello YOLOv9 tramite la libreria `ultralytics`.
