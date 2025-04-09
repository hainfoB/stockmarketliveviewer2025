import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import keras.backend as K
import datetime

# --- Configuration de la page ---
st.set_page_config(page_title="Analyse et Prédiction Boursière par M.Haithem BERKANE AVRIL 2025", layout="wide")
st.title("Analyse et Prédiction Boursière par M.Haithem BERKANE AVRIL 2025")

# Stockage du DataFrame dans st.session_state
if "df" not in st.session_state:
    st.session_state.df = None

# --- Fonction de métrique personnalisée ---
def custom_accuracy(y_true, y_pred):
    # Considère comme correct si l'erreur absolue est <= 10% de la valeur réelle
    diff = K.abs(y_true - y_pred)
    tol = 0.1 * K.abs(y_true)
    correct = K.cast(K.less_equal(diff, tol), K.floatx())
    return K.mean(correct)

# --- Fonctions de chargement et prétraitement ---
@st.cache_data
def load_yfinance_data(ticker, period="5y"):
    try:
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return None

def preprocess_data(df, target_col, seq_length):
    if target_col not in df.columns:
        st.error(f"La colonne {target_col} n'existe pas dans le dataframe.")
        return None, None, None
    data = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return data, scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# --- Construction et optimisation du modèle LSTM ---
def build_model(hp, seq_length):
    model = Sequential()
    lstm_type = hp.Choice('lstm_type', values=['simple', 'bidirectional'])
    units = hp.Int('units', min_value=32, max_value=128, step=16)
    if lstm_type == 'bidirectional':
        model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=(seq_length, 1)))
    else:
        model.add(LSTM(units=units, return_sequences=True, input_shape=(seq_length, 1)))
    dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate1))
    units2 = hp.Int('units2', min_value=16, max_value=64, step=8)
    model.add(LSTM(units=units2))
    dropout_rate2 = hp.Float('dropout_rate2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate2))
    include_dense = hp.Boolean('include_dense')
    if include_dense:
        dense_units = hp.Int('dense_units', min_value=8, max_value=32, step=8)
        model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', custom_accuracy]
    )
    return model

@st.cache_resource
def optimize_lstm_architecture(X_train, y_train, seq_length):
    # Diviser X_train en un sous-ensemble pour validation (20%)
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    def build_model_wrapper(hp):
        return build_model(hp, seq_length)
    tuner = RandomSearch(
        build_model_wrapper,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='lstm_tuning',
        project_name='stock_prediction'
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tuner.search(
        X_train_sub, y_train_sub,
        epochs=30,
        batch_size=64,
        validation_data=(X_val_sub, y_val_sub),
        callbacks=[early_stopping]
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = best_model.fit(
        X_train_sub, y_train_sub,
        epochs=100,
        batch_size=64,
        validation_data=(X_val_sub, y_val_sub),
        callbacks=[early_stopping],
        verbose=1
    )
    return best_model, best_hp, history

# --- Fonctions d'évaluation et de prédiction ---
def make_predictions(model, X, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    diff = np.abs(y_true - y_pred)
    tol = 0.1 * np.abs(y_true)
    acc = np.mean(diff <= tol)
    return rmse, mae, r2, acc

# --- Projection future en ligne (sans utiliser une fonction externe predict_future) ---
# Utilise exactement la même approche que pour les prédictions sur X_test.
def forecast_future(model, last_sequence, scaler, horizon, seq_length):
    future_preds = []
    current_seq = last_sequence.copy()
    for _ in range(horizon):
        pred = model.predict(current_seq.reshape(1, seq_length, 1))[0, 0]
        future_preds.append(pred)
        # Mettre à jour la séquence en décalant d'une position et en ajoutant la prédiction
        current_seq = np.concatenate([current_seq[1:], np.array([[pred]])], axis=0)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    return future_preds

# --- Liste des titres et commodités ---
stock_options = [
    ("Apple Inc.", "AAPL"),
    ("Microsoft Corporation", "MSFT"),
    ("Alphabet Inc.", "GOOGL"),
    ("Amazon.com Inc.", "AMZN"),
    ("Tesla Inc.", "TSLA"),
    ("Netflix Inc.", "NFLX"),
    ("NVIDIA Corporation", "NVDA"),
    ("Meta Platforms Inc.", "FB"),
    ("Alibaba Group", "BABA"),
    ("JPMorgan Chase & Co.", "JPM"),
    ("Contrats à terme sur l'or", "GC=F"),
    ("Contrats à terme sur l'argent", "SI=F"),
    ("Contrats à terme sur le cuivre", "HG=F"),
    ("Contrats à terme sur le platine", "PL=F"),
    ("Contrats à terme sur le palladium", "PA=F"),
    ("Pétrole brut WTI", "CL=F"),
    ("Contrats à terme sur le gaz naturel", "NG=F"),
    ("Contrats à terme sur le maïs", "ZC=F"),
    ("Contrats à terme sur le blé", "ZW=F"),
    ("Contrats à terme sur le soja", "ZS=F"),
    ("Contrats à terme sur le café", "KC=F"),
    ("Contrats à terme sur le sucre", "SB=F"),
    ("Contrats à terme sur le cacao", "CC=F")
]

# --- Fonction principale ---
def main():
    st.sidebar.header("Source des données")
    data_source = st.sidebar.radio("Choisir la source des données", ["Données réelles", "Fichier CSV"])
    df = None
    ticker = None

    if data_source == "Données réelles":
        selected_stock = st.sidebar.selectbox(
            "Sélectionnez le titre ou la commodité",
            stock_options,
            format_func=lambda x: f"{x[0]} ({x[1]})"
        )
        ticker = selected_stock[1]
        period = st.sidebar.selectbox("Période", ["1y", "2y", "5y", "10y"], index=2)
        if st.sidebar.button("Charger les données"):
            with st.spinner("Chargement des données..."):
                st.session_state.df = load_yfinance_data(ticker, period)
    else:
        uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                temp_df = pd.read_csv(uploaded_file)
                date_col = st.sidebar.selectbox("Colonne de date", temp_df.columns)
                if st.sidebar.button("Traiter le CSV"):
                    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                    temp_df.set_index(date_col, inplace=True)
                    st.session_state.df = temp_df
                    ticker = "CSV"
            except Exception as e:
                st.error(f"Erreur lors du chargement du CSV: {str(e)}")

    df = st.session_state.get("df", None)

    if df is not None:
        # Aplatir les colonnes MultiIndex si nécessaire
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

        st.sidebar.header("Paramètres du modèle")
        numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_columns) == 0:
            st.error("Aucune colonne numérique disponible pour la prédiction.")
            return
        target_col = st.sidebar.selectbox("Colonne cible", numeric_columns)
        seq_length = st.sidebar.slider("Longueur des séquences", 1, 100, 30)
        test_size = st.sidebar.slider("Pourcentage de données de test", 0.1, 0.4, 0.2)

        st.subheader("Aperçu des données")
        st.write(df)
        df_reset = df.reset_index()
        x_col = df_reset.columns[0]
        try:
            fig = px.line(df_reset, x=x_col, y=target_col, title=f"Évolution de {target_col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors du tracé du graphique : {str(e)}")

        data, scaled_data, scaler = preprocess_data(df, target_col, seq_length)
        if scaled_data is not None:
            # Création des séquences sur l'ensemble complet
            X_all, y_all = create_sequences(scaled_data, seq_length)
            # Division en deux ensembles : Entraînement et Test
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, shuffle=False)
            n_train_initial = X_train.shape[0]
            # Reshape pour LSTM
            X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
            X_test = X_test.reshape(X_test.shape[0], seq_length, 1)

            col1, col2 = st.columns(2)
            col1.metric("Jours d'entrainement", f"{X_train.shape[0]} échantillons")
            col2.metric("Jours de test", f"{X_test.shape[0]} échantillons")

            if st.button("Réaliser la prédiction"):
                with st.spinner("Optimisation sur l'ensemble d'entraînement..."):
                    best_model, best_hp, history = optimize_lstm_architecture(X_train, y_train, seq_length)
                n_epochs_used = len(history.epoch)
                st.subheader("Hyperparamètres optimaux")
                st.json(best_hp.values)
                st.subheader("Courbes d'apprentissage")
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
                fig_history.update_layout(title='Évolution de la perte (Loss)', xaxis_title='Epochs', yaxis_title='Perte')
                st.plotly_chart(fig_history, use_container_width=True)

                # Prédictions sur l'ensemble d'entraînement et de test
                train_pred = make_predictions(best_model, X_train, scaler)
                test_pred = make_predictions(best_model, X_test, scaler)
                y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
                y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
                train_rmse, train_mae, train_r2, train_acc = calculate_metrics(y_train_inv, train_pred)
                test_rmse, test_mae, test_r2, test_acc = calculate_metrics(y_test_inv, test_pred)
                st.subheader("Métriques d'évaluation")
                metrics_df = pd.DataFrame({
                    'Ensemble': ['Entraînement', 'Test'],
                    'RMSE': [train_rmse, test_rmse],
                    'MAE': [train_mae, test_mae],
                    'R²': [train_r2, test_r2],
                    'Accuracy': [train_acc, test_acc]
                })
                st.table(metrics_df.style.format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'R²': '{:.4f}', 'Accuracy': '{:.2f}'}))

                # Tracé des prédictions vs valeurs réelles
                train_dates = df.index[seq_length:seq_length+len(y_train_inv)]
                test_dates = df.index[-len(y_test_inv):]
                st.subheader("Prédictions vs Valeurs Réelles")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df.index, 
                    y=df[target_col], 
                    mode='lines', 
                    name='Valeurs réelles',
                    line=dict(color='blue')
                ))
                fig_pred.add_trace(go.Scatter(
                    x=train_dates, 
                    y=train_pred.flatten(), 
                    mode='lines', 
                    name="Prédictions d'entraînement",
                    line=dict(color='green')
                ))
                fig_pred.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_pred.flatten(), 
                    mode='lines', 
                    name='Prédictions de test',
                    line=dict(color='red')
                ))
                fig_pred.update_layout(
                    title=f'Prédictions vs Valeurs Réelles pour {target_col}',
                    xaxis_title='Date',
                    yaxis_title=target_col,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # --- Projection future sur l'ensemble complet ---
                st.subheader("Projection future sur l'ensemble complet")
                # Réentraîner le modèle sur l'ensemble complet des données
                X_full, y_full = create_sequences(scaled_data, seq_length)
                X_full = X_full.reshape(X_full.shape[0], seq_length, 1)
                with st.spinner(f"Réentraînement du modèle sur l'ensemble complet pour {n_epochs_used} epochs..."):
                    best_model.fit(X_full, y_full, epochs=n_epochs_used, batch_size=64, verbose=1)
                # Horizon de projection = 30% de la taille de l'ensemble d'entraînement initial
                horizon = int(0.3 * n_train_initial)
                if horizon < 1:
                    horizon = 1
                # Projection future en utilisant une boucle itérative (sans appeler une fonction externe)
                future_preds = []
                current_seq = scaled_data[-seq_length:].copy()
                for _ in range(horizon):
                    pred = best_model.predict(current_seq.reshape(1, seq_length, 1))[0, 0]
                    future_preds.append(pred)
                    current_seq = np.concatenate([current_seq[1:], np.array([[pred]])], axis=0)
                future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                last_date = df.index[-1]
                if isinstance(last_date, (pd.Timestamp, datetime.datetime)):
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
                else:
                    future_dates = pd.RangeIndex(start=int(last_date) + 1, stop=int(last_date) + horizon + 1)
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=df.index[-100:],
                    y=df[target_col][-100:],
                    mode='lines',
                    name='Données historiques',
                    line=dict(color='blue')
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_preds.flatten(),
                    mode='lines',
                    name='Projection future (30% de l\'entraînement)',
                    line=dict(color='red')
                ))
                fig_future.update_layout(
                    title=f'Projection future pour {target_col}',
                    xaxis_title='Date',
                    yaxis_title=target_col,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_future, use_container_width=True)

                future_df = pd.DataFrame({
                    'Date': future_dates,
                    f'Prediction_{target_col}': future_preds.flatten()
                })
                future_df.set_index('Date', inplace=True)
                csv_future = future_df.to_csv()
                st.download_button(
                    label="Télécharger la projection future",
                    data=csv_future,
                    file_name=f"projection_future_{ticker if ticker is not None else 'data'}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
