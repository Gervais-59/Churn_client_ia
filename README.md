

# 📊 Churn Prediction — Pipeline ML de bout en bout

Prédiction du départ (churn) de clients télécom, du preprocessing jusqu'à la mise en production.
Ce projet illustre un **cycle MLOps complet** : entraînement tracké, API de serving, base de données, interface web et déploiement cloud.

> **Démos en ligne**
> 🌐 Interface Streamlit : `https://konan-api-churn-36418387510.europe-west9.run.app/docs`
> ⚙️ API (Swagger) : `https://konan-api-churn-36418387510.europe-west9.run.app/docs`

---

## 🎯 Objectif

Identifier les clients à risque de résiliation à partir de leurs caractéristiques (ancienneté, contrat, services, facturation), afin de permettre des actions de rétention ciblées. Le dataset utilisé est le **Telco Customer Churn** (~7 000 clients, 21 variables).

---

## 🏗️ Architecture

```
┌────────────────────┐      requêtes HTTP      ┌─────────────────────┐
│  Interface Streamlit│ ──────────────────────▶ │  API FastAPI        │
│  (Streamlit Cloud)  │      /predict           │  (Google Cloud Run) │
└────────────────────┘                          └──────────┬──────────┘
                                                            │
                                          ┌─────────────────┴─────────┐
                                          │  Modèle ML (scikit-learn) │
                                          │  + scaler + features      │
                                          └───────────────────────────┘

  Développement / entraînement (local, Docker Compose) :
  ┌──────────┐   ┌────────────┐   ┌───────────────┐
  │ FastAPI  │   │ PostgreSQL │   │  MLflow UI    │
  │ (serving)│   │ (logs préd)│   │ (tracking)    │
  └──────────┘   └────────────┘   └───────────────┘
```

---

## 🧰 Stack technique

| Domaine | Outils |
|---------|--------|
| **Modélisation** | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **Tracking d'expériences** | MLflow (paramètres, métriques, artefacts, courbes ROC) |
| **API / Serving** | FastAPI, Uvicorn, Pydantic |
| **Base de données** | PostgreSQL (logging des prédictions + monitoring) |
| **Conteneurisation** | Docker, Docker Compose |
| **Interface** | Streamlit |
| **Cloud / Déploiement** | Google Cloud Run, Artifact Registry, Cloud Build |
| **Interprétabilité** | SHAP |

---

## 🔬 Démarche de modélisation

### Preprocessing
- Suppression des identifiants et gestion des valeurs manquantes de `TotalCharges`.
- **Label encoding** pour les variables binaires, **one-hot encoding** (avec `drop_first`) pour les multi-catégories afin d'éviter la colinéarité.
- Nettoyage systématique des noms de colonnes pour garantir la cohérence entre entraînement et serving.

### Gestion du déséquilibre (~27 % de churn)
Combinaison **SMOTE** (sur-échantillonnage de la classe minoritaire) + **RandomUnderSampler** (sous-échantillonnage de la majoritaire), appliquée **uniquement sur le train** pour ne pas fausser l'évaluation.

### Feature engineering
Création de variables métier à partir des données brutes, notamment :
- `charge_moyenne` = `TotalCharges / (tenure + 1)` — le coût moyen réel par mois.
- `client_recent` — indicateur des clients de moins de 6 mois, les plus volatils.

### Sélection du modèle
Trois modèles comparés (Logistic Regression, Random Forest, XGBoost) et tous trackés dans MLflow. Le meilleur est sélectionné automatiquement selon le **F1-score** — plus pertinent que l'accuracy sur un jeu déséquilibré.

---

## 🧠 Interprétabilité (SHAP)

L'analyse SHAP révèle les facteurs qui pèsent le plus sur la décision du modèle :

**`tenure` — Ancienneté (facteur n°1)**
L'information la plus déterminante. Un client ancien a des habitudes ancrées ; un client récent a de fortes chances de partir dès les premières insatisfactions.

**`Contract_Two year` — Engagement (facteur n°2)**
Le fait d'avoir un contrat bloqué sur deux ans est décisif : un client engagé reste malgré les insatisfactions.

**`MonthlyCharges` & `InternetService_Fiber optic` — Prix et service**
Une facture élevée pousse au départ (sensibilité aux offres concurrentes). La fibre, contre-intuitivement, est associée à un churn plus élevé : elle coûte plus cher et évolue sur un marché où les concurrents multiplient les promotions agressives.

**Écart entre charge moyenne et facture actuelle**
En comparant `charge_moyenne` (historique) et `MonthlyCharges` (actuel) :
- **Divergence à la hausse** → forte incitation au départ. Souvent le signe de la fin d'une promotion ou de frais ajoutés : cette mauvaise surprise financière pousse le client à comparer la concurrence.
- **Prix stable** → fidélisation. Une facturation prévisible installe le client dans une routine.
- **Divergence à la baisse** → fidélisation renforcée (remise de fidélité, downgrade).

---

## 📈 Monitoring & production

L'API expose des endpoints de suivi qui, en production, permettent de détecter le **model drift** :
- `/predict` — prédiction unitaire + logging automatique en PostgreSQL.
- `/predictions` — historique des prédictions.
- `/predictions/stats` — statistiques agrégées (taux de churn prédit, probabilité moyenne).

Le logging systématique des prédictions constitue la base d'un suivi de performance dans le temps.

---

## 🚀 Lancement

### En local (stack complète avec Docker Compose)

```bash
docker compose up --build
```

| Service | URL locale |
|---------|-----------|
| API (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| PostgreSQL | localhost:5432 |

### Interface Streamlit (local)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📁 Structure du projet

```
.
├── train.py                # Entraînement + tracking MLflow
├── app.py                  # API FastAPI (version complète avec PostgreSQL)
├── app_gcp.py              # API FastAPI (version Cloud Run, sans BDD)
├── streamlit_app.py        # Interface web
├── docker-compose.yml      # Orchestration API + PostgreSQL + MLflow
├── Dockerfile              # Image de l'API
├── requirements.txt
└── README.md
```

---

## 👤 Auteur

**Konan Gervais N'Guessan** — Data Scientist
📧 konangervaisn@gmail.com · [LinkedIn](https://linkedin.com/in/konan-gervais-n-guessan) · [GitHub](https://github.com/Gervais-59)
