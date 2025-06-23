import streamlit as st
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Initialisierung des Session State
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'confidences' not in st.session_state:
    st.session_state.confidences = []
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = []
if 'markov_chain' not in st.session_state:
    st.session_state.markov_chain = defaultdict(lambda: defaultdict(int))


def create_features(sequence, window_size=5):
    """Erstellt Merkmale aus der Sequenz für das ML-Modell"""
    if len(sequence) < window_size:
        return None

    features = []
    for i in range(window_size, len(sequence)):
        window = sequence[i - window_size:i]
        feature_row = window.copy()
        feature_row.extend([
            sum(window),
            np.mean(window),
            len([j for j in range(1, len(window)) if window[j] != window[j - 1]]),
            window[-1],
            window.count(1),
            window.count(0),
        ])
        features.append(feature_row)
    return np.array(features)


def update_markov_chain(sequence):
    """Aktualisiert die Markow-Kette mit neuen Sequenzdaten"""
    if len(sequence) < 2:
        return
    for i in range(len(sequence) - 1):
        current = sequence[i]
        next_val = sequence[i + 1]
        st.session_state.markov_chain[current][next_val] += 1


def predict_next_markov(sequence):
    """Sagt den nächsten Wert mit der Markow-Kette voraus"""
    if len(sequence) == 0:
        return 0.5, 0  # Zufällige Schätzung

    last_val = sequence[-1]
    counts = st.session_state.markov_chain[last_val]

    if sum(counts.values()) == 0:
        return 0.5, 0  # Keine Daten, Zufall

    total = sum(counts.values())
    prob_1 = counts[1] / total
    prob_0 = counts[0] / total

    if prob_1 > prob_0:
        return prob_1, 1
    else:
        return prob_0, 0


def predict_next_ml(sequence, window_size=5):
    """Sagt den nächsten Wert mit dem ML-Modell voraus"""
    if len(sequence) < window_size + 1:
        return 0.5, 0  # Nicht genug Daten

    features = create_features(sequence, window_size)
    if features is None or len(features) < 2:
        return 0.5, 0

    targets = sequence[window_size:]
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
    try:
        model.fit(features, targets)
        last_window = sequence[-window_size:]
        pred_features = last_window.copy()
        pred_features.extend([
            sum(last_window),
            np.mean(last_window),
            len([j for j in range(1, len(last_window)) if last_window[j] != last_window[j - 1]]),
            last_window[-1],
            last_window.count(1),
            last_window.count(0),
        ])
        pred_features = np.array(pred_features).reshape(1, -1)
        prediction = model.predict(pred_features)[0]
        probabilities = model.predict_proba(pred_features)[0]
        confidence = max(probabilities)
        return confidence, prediction
    except:
        return 0.5, 0

# Streamlit UI
st.title("Be Random - Vorhersage menschlicher Zufallsfolgen")
st.markdown(
    "Versuchen Sie, eine zufällige Folge aus Nullen und Einsen einzugeben. Die KI wird versuchen, Ihre Muster zu erkennen und den nächsten Wert vorherzusagen."
)

# Modellauswahl
model_type = st.selectbox("Vorhersagemodell auswählen:", ["Maschinelles Lernen", "Markow-Kette", "Beides"])

# Eingabebereich
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("0 eingeben", key="btn_0", use_container_width=True):
        new_input = 0
        st.session_state.sequence.append(new_input)
        if len(st.session_state.sequence) > 1:
            if model_type == "Maschinelles Lernen":
                conf, pred = predict_next_ml(st.session_state.sequence[:-1])
            elif model_type == "Markow-Kette":
                conf, pred = predict_next_markov(st.session_state.sequence[:-1])
            else:
                conf_ml, pred_ml = predict_next_ml(st.session_state.sequence[:-1])
                conf_mc, pred_mc = predict_next_markov(st.session_state.sequence[:-1])
                if conf_ml > conf_mc:
                    conf, pred = conf_ml, pred_ml
                else:
                    conf, pred = conf_mc, pred_mc

            st.session_state.predictions.append(pred)
            st.session_state.confidences.append(conf)
            st.session_state.correct_predictions.append(pred == new_input)

        update_markov_chain(st.session_state.sequence)

with col2:
    if st.button("1 eingeben", key="btn_1", use_container_width=True):
        new_input = 1
        st.session_state.sequence.append(new_input)
        if len(st.session_state.sequence) > 1:
            if model_type == "Maschinelles Lernen":
                conf, pred = predict_next_ml(st.session_state.sequence[:-1])
            elif model_type == "Markow-Kette":
                conf, pred = predict_next_markov(st.session_state.sequence[:-1])
            else:
                conf_ml, pred_ml = predict_next_ml(st.session_state.sequence[:-1])
                conf_mc, pred_mc = predict_next_markov(st.session_state.sequence[:-1])
                if conf_ml > conf_mc:
                    conf, pred = conf_ml, pred_ml
                else:
                    conf, pred = conf_mc, pred_mc

            st.session_state.predictions.append(pred)
            st.session_state.confidences.append(conf)
            st.session_state.correct_predictions.append(pred == new_input)

        update_markov_chain(st.session_state.sequence)

with col3:
    if st.button("Zurücksetzen", key="reset"):
        st.session_state.sequence = []
        st.session_state.predictions = []
        st.session_state.confidences = []
        st.session_state.correct_predictions = []
        st.session_state.markov_chain = defaultdict(lambda: defaultdict(int))
        st.rerun()

# Aktuelle Sequenz anzeigen
if st.session_state.sequence:
    st.subheader("Ihre aktuelle Sequenz:")
    sequence_display = " → ".join(map(str, st.session_state.sequence))
    st.code(sequence_display)

    # Statistiken
    if len(st.session_state.correct_predictions) > 0:
        accuracy = np.mean(st.session_state.correct_predictions) * 100
        avg_confidence = np.mean(st.session_state.confidences) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Länge der Sequenz", len(st.session_state.sequence))
        with col2:
            st.metric("Vorhersagegenauigkeit", f"{accuracy:.1f}%")
        with col3:
            st.metric("Durchschnittliche Sicherheit", f"{avg_confidence:.1f}%")

# Visualisierung
if len(st.session_state.predictions) > 0:
    st.subheader("Vorhersage-Leistung")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Sequenz & Vorhersagen', 'Vorhersage-Sicherheit', 'Kumulative Genauigkeit'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    x_vals = list(range(1, len(st.session_state.predictions) + 1))
    actual_vals = st.session_state.sequence[1:]

    fig.add_trace(
        go.Scatter(x=x_vals, y=actual_vals, mode='lines+markers',
                   name='Tatsächlich', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_vals, y=st.session_state.predictions, mode='lines+markers',
                   name='Vorhersage', line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    colors = ['green' if correct else 'red' for correct in st.session_state.correct_predictions]
    fig.add_trace(
        go.Bar(x=x_vals, y=st.session_state.confidences,
               name='Sicherheit', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    cumulative_accuracy = np.cumsum(st.session_state.correct_predictions) / np.arange(1, len(st.session_state.correct_predictions) + 1)
    fig.add_trace(
        go.Scatter(x=x_vals, y=cumulative_accuracy, mode='lines+markers',
                   name='Kumulative Genauigkeit', line=dict(color='purple', width=3)),
        row=3, col=1
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="Zufall (50%)", row=3, col=1)

    fig.update_layout(height=800, showlegend=True, title_text="Analyse der KI-Vorhersagen")
    fig.update_xaxes(title_text="Vorhersagenummer", row=3, col=1)
    fig.update_yaxes(title_text="Wert", row=1, col=1)
    fig.update_yaxes(title_text="Sicherheit", row=2, col=1)
    fig.update_yaxes(title_text="Genauigkeit", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Nächste Vorhersage
    if len(st.session_state.sequence) > 0:
        st.subheader("Nächste Vorhersage")
        if model_type == "Maschinelles Lernen":
            next_conf, next_pred = predict_next_ml(st.session_state.sequence)
            model_used = "ML"
        elif model_type == "Markow-Kette":
            next_conf, next_pred = predict_next_markov(st.session_state.sequence)
            model_used = "Markow"
        else:
            conf_ml, pred_ml = predict_next_ml(st.session_state.sequence)
            conf_mc, pred_mc = predict_next_markov(st.session_state.sequence)
            if conf_ml > conf_mc:
                next_conf, next_pred = conf_ml, pred_ml
                model_used = "ML"
            else:
                next_conf, next_pred = conf_mc, pred_mc
                model_used = "Markow"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vorhergesagter nächster Wert", int(next_pred))
        with col2:
            st.metric("Sicherheit", f"{next_conf:.1%}")
        with col3:
            st.metric("Verwendetes Modell", model_used)
else:
    st.info(
        "Starten Sie, indem Sie entweder '0' oder '1' klicken, um Ihre Sequenz zu beginnen. Nach der zweiten Eingabe beginnt die KI mit der Vorhersage."
    )

# Anleitung
with st.expander("Wie funktioniert es?"):
    st.markdown("""
    **Maschinelles Lernmodell**: Verwendet einen Random Forest-Klassifikator, der Muster in Ihren letzten Eingaben erkennt, 
    einschließlich Sequenzen, Häufigkeiten und Wechsel.

    **Markow-Kettenmodell**: Verwendet einfache Wahrscheinlichkeiten, basierend darauf, was typischerweise auf einen bestimmten Wert folgt.

    **Warum funktioniert das?**: Menschlich erzeugte "Zufallsfolgen" enthalten oft unbewusste Muster, Wiederholungen 
    und Vorlieben, die von maschinellen Lernverfahren erkannt und genutzt werden können.

    **Versuchen Sie, wirklich zufällig zu sein** - das ist schwieriger als man denkt!
    """)
