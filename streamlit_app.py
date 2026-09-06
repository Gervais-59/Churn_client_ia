"""
streamlit_app.py — Interface web pour l'API de prédiction de churn.
La façade appelle l'API FastAPI déployée sur Google Cloud Run.

Lancement local :  streamlit run streamlit_app.py
"""
import requests
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# ⚠️ mon URL Cloud Run (sans slash final)
API_URL = "https://konan-api-churn-36418387510.europe-west9.run.app"

st.set_page_config(page_title="Prédiction de Churn", page_icon="📊", layout="centered")

st.title("📊 Prédiction de Churn Client")
st.caption("Démo — modèle de ML servi par une API FastAPI sur Google Cloud Run")

st.divider()

#
# FORMULAIRE — variables métier parlantes

st.subheader("Profil du client")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    monthly_charges = st.slider("Charges mensuelles (€)", 18.0, 120.0, 65.0)
    contract = st.selectbox(
        "Type de contrat",
        ["Mensuel", "1 an", "2 ans"],
    )
    internet = st.selectbox(
        "Service Internet",
        ["Fibre optique", "DSL", "Aucun"],
    )
    payment = st.selectbox(
        "Mode de paiement",
        ["Chèque électronique", "Chèque postal", "Virement bancaire", "Carte de crédit"],
    )

with col2:
    senior = st.checkbox("Client senior (65+)")
    partner = st.checkbox("En couple")
    dependents = st.checkbox("Personnes à charge")
    online_security = st.checkbox("Sécurité en ligne")
    tech_support = st.checkbox("Support technique")

st.divider()


# CONSTRUCTION DU PAYLOAD (traduction métier → format modèle)

def build_payload():
    # Ancienneté récente ?
    client_recent = 1.0 if tenure < 6 else 0.0
    # Charge moyenne approximée (le vrai calcul TotalCharges/tenure)
    total_charges = monthly_charges * max(tenure, 1)
    charge_moyenne = total_charges / (tenure + 1)

    # One-hot du contrat (Mensuel = référence, donc 0/0)
    contract_one_year = 1.0 if contract == "1 an" else 0.0
    contract_two_year = 1.0 if contract == "2 ans" else 0.0

    # One-hot Internet (DSL = référence)
    internet_fiber = 1.0 if internet == "Fibre optique" else 0.0
    internet_no = 1.0 if internet == "Aucun" else 0.0

    # One-hot paiement (Virement = référence)
    pay_cc = 1.0 if payment == "Carte de crédit" else 0.0
    pay_echeck = 1.0 if payment == "Chèque électronique" else 0.0
    pay_mail = 1.0 if payment == "Chèque postal" else 0.0

    return {
        "gender": 0.0,
        "senior_citizen": float(senior),
        "partner": float(partner),
        "dependents": float(dependents),
        "tenure": float(tenure),
        "phone_service": 1.0,
        "online_security": float(online_security),
        "online_backup": 0.0,
        "device_protection": 0.0,
        "tech_support": float(tech_support),
        "streaming_tv": 0.0,
        "streaming_movies": 0.0,
        "paperless_billing": 1.0,
        "monthly_charges": float(monthly_charges),
        "total_charges": float(total_charges),
        "multiple_lines_no_phone_service": 0.0,
        "multiple_lines_yes": 0.0,
        "internet_service_fiber_optic": internet_fiber,
        "internet_service_no": internet_no,
        "contract_one_year": contract_one_year,
        "contract_two_year": contract_two_year,
        "payment_method_credit_card_automatic": pay_cc,
        "payment_method_electronic_check": pay_echeck,
        "payment_method_mailed_check": pay_mail,
        "charge_moyenne": charge_moyenne,
        "client_recent": client_recent,
    }



# PRÉDICTION

if st.button("🔮 Prédire le churn", type="primary", use_container_width=True):
    payload = build_payload()

    try:
        with st.spinner("Appel de l'API..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

        proba = result["proba_churn"]
        prediction = result["prediction"]

        st.divider()

        # Jauge visuelle
        st.subheader("Résultat")
        st.progress(proba)
        st.metric("Probabilité de churn", f"{proba:.1%}")

        # Verdict coloré
        if prediction == "churn":
            st.error(f"⚠️ Client À RISQUE — probabilité de départ : {proba:.1%}")
        else:
            st.success(f"✅ Client STABLE — probabilité de départ : {proba:.1%}")

        # Interprétation du profil
        st.subheader("Analyse du profil")
        raisons = []
        if contract == "Mensuel":
            raisons.append("📄 Contrat mensuel (sans engagement) — facteur de churn majeur")
        if internet == "Fibre optique":
            raisons.append("🌐 Fibre optique — segment au churn élevé")
        if payment == "Chèque électronique":
            raisons.append("💳 Paiement par chèque électronique — corrélé au churn")
        if tenure < 6:
            raisons.append("🆕 Client récent — pas encore fidélisé")
        if not online_security and not tech_support:
            raisons.append("🔓 Aucun service additionnel — faible ancrage")

        if raisons:
            st.write("**Signaux de risque détectés :**")
            for r in raisons:
                st.write(f"- {r}")
        else:
            st.write("Profil sans signal de risque majeur — client bien ancré.")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        st.info("Vérifie que l'URL de l'API est correcte et que le service Cloud Run est actif.")

st.divider()
st.caption("Konan Gervais N'Guessan — Projet MLOps Churn Prediction")