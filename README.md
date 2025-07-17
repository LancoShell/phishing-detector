# Advanced Automated Phishing Email Detector

Uno script Python avanzato per rilevare email di phishing usando NLP e machine learning.

---

## Funzionalità

- Pulizia e preprocessamento avanzato testo (lemmatizzazione, rimozione stopwords, masking email/URL/numeri)
- Pipeline con TF-IDF, riduzione dimensionale (PCA), e classificatore Random Forest
- Ricerca iperparametri tramite GridSearchCV
- Report di valutazione dettagliato con accuracy, ROC-AUC, confusion matrix, classification report
- Salvataggio e caricamento modello con joblib
- Modalità di training, valutazione e predizione da linea di comando

---

## Requisiti

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- joblib

Installa con:

```bash
pip install requirements.txt
```


