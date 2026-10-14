import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- CONFIGURAZIONE PAGINA WEB ---
st.set_page_config(page_title="Analisi Voti Forum", layout="wide", page_icon="📊")
st.title("📊 Dashboard Analisi Voti Discourse")

# --- FUNZIONI DI CALCOLO IN CACHE ---
@st.cache_data
def carica_dati(percorso_file):
    return pd.read_csv(percorso_file)

@st.cache_data
def calcola_matrici_base(df):
    matrice_voti = df.pivot(index='Utente', columns='Round_Poll', values='Voto')
    totale_voti_per_utente = matrice_voti.notna().sum(axis=1)
    utenti = matrice_voti.index
    
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
                voti_uguali = (matrice_voti.loc[u1][poll_comuni] == matrice_voti.loc[u2][poll_comuni]).sum()
                matrice_grezza.loc[u1, u2] = float(voti_uguali / totale_comuni)
                matrice_confidenza.loc[u1, u2] = float(totale_comuni / totale_voti_per_utente.loc[u1])
                
    return matrice_grezza, matrice_confidenza

@st.cache_data
def calcola_metriche_community(df):
    # 1. PARTECIPAZIONE
    presenze = df.groupby('Utente')['Round_Poll'].nunique().reset_index()
    presenze.columns = ['Utente', 'Sondaggi_Votati']
    presenze = presenze.sort_values(by='Sondaggi_Votati', ascending=False)
    
    # 2. COERENZA
    df['Tema'] = df['Round_Poll'].apply(lambda x: x.rsplit('_', 1)[0] if '_' in x else x)
    voti_per_tema = df.groupby(['Utente', 'Tema'])['Voto'].nunique().reset_index()
    voti_per_tema['È_Coerente'] = voti_per_tema['Voto'] == 1
    coerenza = voti_per_tema.groupby('Utente')['È_Coerente'].mean().reset_index()
    coerenza.columns = ['Utente', 'Tasso_Coerenza_%']
    coerenza['Tasso_Coerenza_%'] = (coerenza['Tasso_Coerenza_%'] * 100).round(1)
    coerenza = coerenza.sort_values(by='Tasso_Coerenza_%', ascending=False)
    
    # 3. CONFORMISMO
    totale_voti_poll = df.groupby('Round_Poll')['Utente'].count().reset_index(name='Totale_Voti')
    voti_opzione = df.groupby(['Round_Poll', 'Voto'])['Utente'].count().reset_index(name='Voti_Ricevuti')
    popolarita = pd.merge(voti_opzione, totale_voti_poll, on='Round_Poll')
    popolarita['Popolarita_Scelta_%'] = (popolarita['Voti_Ricevuti'] / popolarita['Totale_Voti']) * 100
    df_con_pesi = pd.merge(df, popolarita[['Round_Poll', 'Voto', 'Popolarita_Scelta_%']], on=['Round_Poll', 'Voto'])
    conformismo = df_con_pesi.groupby('Utente')['Popolarita_Scelta_%'].mean().reset_index()
    conformismo['Popolarita_Scelta_%'] = conformismo['Popolarita_Scelta_%'].round(1)
    
    return presenze, coerenza, conformismo

# --- CARICAMENTO DATI ---
try:
    df_voti = carica_dati('voti_discourse.csv')
except FileNotFoundError:
    st.error("Errore: Impossibile trovare 'voti_discourse.csv'. Assicurati che il file sia nella stessa cartella.")
    st.stop()

with st.spinner("Elaborazione dei dati in corso..."):
    matrice_grezza, matrice_confidenza = calcola_matrici_base(df_voti)
    df_presenze, df_coerenza, df_conformismo = calcola_metriche_community(df_voti)

# Determina il numero massimo di voti per lo slider
max_voti = int(df_presenze['Sondaggi_Votati'].max()) if not df_presenze.empty else 10

# --- SIDEBAR INTERATTIVA ---
st.sidebar.header("⚙️ Mappa Affinità")

peso_confidenza = st.sidebar.slider(
    "Peso della Confidenza (Penalità per pochi voti):", 
    min_value=0.0, max_value=2.0, value=1.0, step=0.1,
    help="0.0 = Affinità grezza. > 1.0 = Chi vota poco perde molta affinità."
)

soglia = st.sidebar.slider(
    "Nascondi affinità finale sotto il:", 
    min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.header("🏆 Classifiche Community")

# NUOVO SLIDER: Soglia minima di voti per entrare in classifica
min_voti_classifica = st.sidebar.slider(
    "Minimo sondaggi votati per entrare in classifica:", 
    min_value=1, max_value=max_voti, value=3, step=1,
    help="Esclude dalle metriche di Coerenza e Mainstream gli utenti che hanno partecipato troppo poco per essere valutati."
)

# --- NAVIGAZIONE A SCHEDE (TABS) ---
tab1, tab2 = st.tabs(["🤝 Affinità e Legami", "🏆 Classifiche Community"])

# === TAB 1: MATRICE E LEGAMI ===
with tab1:
    matrice_ponderata = matrice_grezza * (matrice_confidenza ** peso_confidenza)
    matrice_filtrata = matrice_ponderata.where(matrice_ponderata >= soglia)

    st.subheader(f"Heatmap di Affinità (Peso Confidenza: {peso_confidenza})")
    
    fig_heatmap = px.imshow(
        matrice_filtrata,
        labels=dict(x="Utente B (Valutato)", y="Utente A (Soggetto)", color="Affinità Ponderata"),
        x=matrice_filtrata.columns,
        y=matrice_filtrata.index,
        color_continuous_scale="Inferno", 
        zmin=0, zmax=1
    )
    fig_heatmap.update_layout(width=800, height=700)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Esplora le Anime Gemelle")
    utente_selezionato = st.selectbox("Dal punto di vista di quale utente vuoi guardare?", matrice_ponderata.index)

    simili = matrice_ponderata.loc[utente_selezionato].drop(utente_selezionato).dropna()

    if not simili.empty:
        top_5 = simili.sort_values(ascending=False).head(5)
        st.write(f"I profili più affini a **{utente_selezionato}** sono:")
        for utente, punteggio in top_5.items():
            st.progress(float(punteggio), text=f"{utente} (Punteggio ponderato: {punteggio*100:.1f}%)")
    else:
        st.warning("Nessun dato sufficiente per calcolare le affinità di questo utente.")


# === TAB 2: LEADERBOARD ===
with tab2:
    st.subheader("Le metriche comportamentali della community")
    
    # Applichiamo il filtro del nuovo slider: troviamo chi ha raggiunto il limite minimo di voti
    utenti_validi = df_presenze[df_presenze['Sondaggi_Votati'] >= min_voti_classifica]['Utente']
    
    # Filtriamo i DataFrame di Coerenza e Conformismo
    df_coerenza_filtrato = df_coerenza[df_coerenza['Utente'].isin(utenti_validi)]
    df_conformismo_filtrato = df_conformismo[df_conformismo['Utente'].isin(utenti_validi)]
    
    # 3 Colonne per le classifiche testuali
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("🏃 **I più Attivi** (Sondaggi votati)")
        st.dataframe(df_presenze, hide_index=True, use_container_width=True)
        
    with col2:
        st.write(f"🛡️ **I più Coerenti** (Min. {min_voti_classifica} voti)")
        st.dataframe(df_coerenza_filtrato, hide_index=True, use_container_width=True)
        
    with col3:
        st.write(f"🐑 **I più Mainstream** (Min. {min_voti_classifica} voti)")
        df_conformismo_ordinato = df_conformismo_filtrato.sort_values(by='Popolarita_Scelta_%', ascending=False)
        st.dataframe(df_conformismo_ordinato, hide_index=True, use_container_width=True)

    st.markdown("---")
    
    # Grafico a dispersione: Partecipazione vs Conformismo
    st.subheader(f"Analisi: Veterani vs Mainstream (Solo utenti con ≥ {min_voti_classifica} voti)")
    df_analisi = pd.merge(df_presenze[df_presenze['Utente'].isin(utenti_validi)], df_conformismo_filtrato, on='Utente')
    
    # Controlliamo che ci siano dati prima di disegnare il grafico (evita errori se il filtro è troppo alto)
    if not df_analisi.empty:
        fig_scatter = px.scatter(
            df_analisi, 
            x="Sondaggi_Votati", 
            y="Popolarita_Scelta_%",
            text="Utente",
            labels={
                "Sondaggi_Votati": "Numero di Partecipazioni",
                "Popolarita_Scelta_%": "Tendenza Mainstream (%)"
            },
            title="I veri Anticonformisti si trovano in basso (votano le nicchie)"
        )
        fig_scatter.update_traces(textposition='top center')
        # Aggiungiamo linee mediane per dividere in quadranti
        fig_scatter.add_vline(x=df_analisi['Sondaggi_Votati'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_hline(y=df_analisi['Popolarita_Scelta_%'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Nessun utente soddisfa i criteri del filtro per mostrare il grafico.")
