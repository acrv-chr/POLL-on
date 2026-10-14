import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Analisi Voti Forum", layout="wide")
st.title("📊 Dashboard Analisi Voti Discourse")

# --- MOTORE DI CALCOLO MATRICI ---
@st.cache_data
def calcola_matrici_base(percorso_file):
    df = pd.read_csv(percorso_file)
    matrice_voti = df.pivot(index='Utente', columns='Round_Poll', values='Voto')
    
    # Calcoliamo quanti sondaggi ha votato ogni utente in totale
    totale_voti_per_utente = matrice_voti.notna().sum(axis=1)
    
    utenti = matrice_voti.index
    # Ora calcoliamo DUE matrici: quella dell'accordo grezzo e quella della confidenza
    matrice_grezza = pd.DataFrame(index=utenti, columns=utenti, dtype=float)
    matrice_confidenza = pd.DataFrame(index=utenti, columns=utenti, dtype=float)
    
    for u1 in utenti:
        for u2 in utenti:
            poll_comuni = matrice_voti.loc[u1].notna() & matrice_voti.loc[u2].notna()
            totale_comuni = poll_comuni.sum()
            
            if totale_comuni == 0:
                matrice_grezza.loc[u1, u2] = np.nan
                matrice_confidenza.loc[u1, u2] = np.nan
            else:
                # 1. Accordo grezzo (Su quelli in comune, quante volte coincidono?)
                voti_uguali = (matrice_voti.loc[u1][poll_comuni] == matrice_voti.loc[u2][poll_comuni]).sum()
                matrice_grezza.loc[u1, u2] = float(voti_uguali / totale_comuni)
                
                # 2. Confidenza (Rispetto a tutto ciò che u1 ha votato, quanto u2 era presente?)
                matrice_confidenza.loc[u1, u2] = float(totale_comuni / totale_voti_per_utente.loc[u1])
                
    return matrice_grezza, matrice_confidenza

# Caricamento e pre-calcolo
try:
    with st.spinner("Calcolo delle matrici in corso..."):
        matrice_grezza, matrice_confidenza = calcola_matrici_base('voti_discourse.csv')
except FileNotFoundError:
    st.error("Errore: Impossibile trovare 'voti_discourse.csv'.")
    st.stop()

# --- INTERFACCIA WEB (SIDEBAR) ---
st.sidebar.header("⚙️ Impostazioni Algoritmo")

# NUOVO SLIDER: Il potere del coefficiente di confidenza!
peso_confidenza = st.sidebar.slider(
    "Peso della Confidenza (Penalità per pochi voti):", 
    min_value=0.0, 
    max_value=2.0, 
    value=1.0, 
    step=0.1,
    help="0.0 = Nessuna penalità (Affinità grezza). 1.0 = Penalità lineare (chi vota poco perde molta affinità). > 1.0 = Penalizzazione severa."
)

soglia = st.sidebar.slider(
    "Nascondi affinità finale sotto il:", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.0, 
    step=0.05
)

# --- APPLICAZIONE DELLA FORMULA MATEMATICA IN TEMPO REALE ---
# Affinità Ponderata = Affinità Grezza * (Confidenza ^ Peso)
matrice_ponderata = matrice_grezza * (matrice_confidenza ** peso_confidenza)

# Filtro visivo
matrice_filtrata = matrice_ponderata.where(matrice_ponderata >= soglia)

# --- VISUALIZZAZIONE ---
st.subheader(f"1. Heatmap di Affinità (Peso Confidenza: {peso_confidenza})")
st.write("Le righe rappresentano il punto di vista dell'Utente A. Nota come ora la mappa sia asimmetrica!")

fig_heatmap = px.imshow(
    matrice_filtrata,
    labels=dict(x="Utente B (Che viene giudicato)", y="Utente A (Che giudica)", color="Affinità Ponderata"),
    x=matrice_filtrata.columns,
    y=matrice_filtrata.index,
    color_continuous_scale="Inferno", # Messo Inferno per contrastare coi valori nulli scuri
    zmin=0, zmax=1
)
fig_heatmap.update_layout(width=800, height=800)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.subheader("2. Elenco Affinità")
utente_selezionato = st.selectbox("Dal punto di vista di quale utente?", matrice_ponderata.index)

# Leggiamo la riga specifica (quindi come l'utente_selezionato vede gli altri)
simili = matrice_ponderata.loc[utente_selezionato].drop(utente_selezionato).dropna()

if not simili.empty:
    top_5 = simili.sort_values(ascending=False).head(5)
    st.write(f"Considerando sia le scelte fatte che il volume di partecipazione, i profili più affini a **{utente_selezionato}** sono:")
    for utente, punteggio in top_5.items():
        st.progress(float(punteggio), text=f"{utente} (Punteggio ponderato: {punteggio*100:.1f}%)")
else:
    st.write("Nessun dato sufficiente per questo utente.")