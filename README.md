# ScentMatch — application de démonstration

Application Streamlit de prédiction de la famille olfactive d'une description. Le modèle SVM et le pipeline TF-IDF sont chargés depuis `artifacts/scentmatch_model.joblib`. Les sept probabilités affichées proviennent du modèle fourni.

## Lancer localement

Python 3.13 est utilisé pour cette version. Depuis la racine de ce dossier :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publier sur Streamlit Community Cloud

1. Créez un dépôt GitHub et ajoutez le contenu de `DEPLOI_SCENTMATCH` à sa racine. N'ajoutez pas de fichier `secrets.toml` ni d'informations privées.
2. Poussez le dépôt vers GitHub avec votre compte.
3. Dans [Streamlit Community Cloud](https://share.streamlit.io/), connectez GitHub et choisissez **Create app**.
4. Sélectionnez le dépôt, la branche, `app.py`, puis, dans les paramètres avancés, choisissez Python **3.13**.
5. Cliquez sur **Deploy**. La page de gestion affichera l'URL publique `https://<sous-domaine>.streamlit.app` à transmettre à votre professeure.

Streamlit Cloud installe les dépendances déclarées dans `requirements.txt` sur Linux. Les versions de scikit-learn, NumPy, SciPy et joblib sont épinglées pour rester compatibles avec le modèle sérialisé. `.python-version` sert de repère dans le dépôt ; sélectionnez explicitement Python 3.13 dans les paramètres Cloud.

Le dépôt préparé localement n'a pas été publié. La publication sur GitHub et le déploiement Cloud restent à effectuer depuis votre compte.
