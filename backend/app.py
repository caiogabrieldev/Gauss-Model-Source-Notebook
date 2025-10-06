from flask import Flask, request, jsonify
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from flask_cors import CORS
import io # Used for reading CSV from memory


# Segment Tree Class
class SegmentTree:
    # Segment Tree for cumulative sum computation
    def __init__(self, arr):
        self.n = 1
        while self.n < len(arr):
            self.n *= 2
        self.tree = [0] * (2 * self.n)
        for i, v in enumerate(arr):
            self.tree[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    # Query cumulative sum from index 0 to i
    def query(self, i):
        # Query cumulative sum from index 0 to i (inclusive)
        # Note: The original SegmentTree implementation was slightly non-standard.
        # Given the usage `sum(self.tree[self.n: self.n + i + 1])`,
        # it sums the leaf nodes directly up to the desired index.
        # This implementation avoids the complex log(n) query.
        return sum(self.tree[self.n: self.n + i + 1])


# Flask App Initialization
app = Flask(__name__)
CORS(app)

# --- Model Loading (Assumes files exist in ./model/) ---
try:
    # Load trained Autoencoder
    mlp_ae = joblib.load("./model/autoencoder_modelv4.pkl")

    # Load trained XGBoost model
    bst_ae = xgb.Booster()
    bst_ae.load_model("./model/xgb_model_aev4.json")

    print(f"Model was trained with num_class: {bst_ae.attr('num_class')}")
    # Load feature order used by XGBoost
    FEATURES_XGB = joblib.load("./model/features_xgb.pkl")

    model_dump = bst_ae.get_dump(dump_format='json')
    import json
    config = json.loads(model_dump[0]) if model_dump else {}
    print(f"Model was trained with num_class: {bst_ae.attr('koi_disposition')}")
    # Load dataset to calculate column means for missing features
    # NOTE: The 'normalized_data.xls' path suggests an Excel file, but uses 'pd.read_csv'.
    # I'll keep the original call assuming the file type is correct or a placeholder.
    df = pd.read_csv("normalized_data.xls", comment='#')
    print(df['koi_disposition'])
    X = df.drop(columns=['koi_disposition'])
    # y = df['koi_disposition'].astype('category').cat.codes # Ensure y is numerical for feature selection
    y = df['koi_disposition']

    print("Unique classes:", sorted(y.unique()))
    print("Class counts:", y.value_counts().sort_index())

    # Split dataset for feature selection (train/test not used for prediction here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # FIX: Changing k=21 to k=20. This selects 20 features, and after removing 'koi_score',
    # we are left with 19 features, matching the 19 expected by MLPRegressor.
    selector = SelectKBest(score_func=f_regression, k=20)
    selector.fit(X_train, y_train)

    # Keep only top features excluding 'koi_score'
    selected_features = X_train.columns[selector.get_support()]
    selected_features_sem_score = [f for f in selected_features if f != 'koi_score']

    # Define continuous columns for cumulative features and Autoencoder
    continuous_cols = selected_features_sem_score # Should now have 19 features

    # Calculate mean values for continuous columns for missing feature handling
    col_means = df[continuous_cols].mean()

except Exception as e:
    print(f"ERROR LOADING ML ARTIFACTS OR DATA: {e}")
    # Initialize placeholders to prevent immediate crash if files are missing
    mlp_ae = None
    bst_ae = None
    FEATURES_XGB = []
    y_test = pd.Series([]) # Empty Series
    continuous_cols = []
    col_means = {}

# Load Kepler dataset for the /get-planets endpoint
try:
    planets = pd.read_csv('kepler_data.csv', comment="#", on_bad_lines='skip')
except Exception as e:
    print(f"ERROR LOADING kepler_data.csv: {e}")
    planets = pd.DataFrame()


# Preprocessing Function
def preprocess_input(input_df):
    """
    Applies feature engineering (cumulative sum, reconstruction error) and
    ensures the final DataFrame has the exact column structure (features and order)
    required by the XGBoost model (FEATURES_XGB).
    """
    input_df = input_df.copy()

    # 1. Fill missing values in base features required for engineering
    for col in continuous_cols:
        if col in input_df.columns:
            # Only fill if the mean is available
            if col in col_means:
                 input_df[col] = input_df[col].fillna(col_means[col])
            # If the column exists but mean is not available, we can drop it later

    # 2. Compute cumulative features using Segment Tree
    for col in continuous_cols:
        if col in input_df.columns:
            # Check if the feature engineering results in a feature name the model expects
            cum_col_name = f'{col}_cum'
            if cum_col_name in FEATURES_XGB:
                # Use .to_numpy() which is preferred over .values for consistency
                st = SegmentTree(input_df[col].fillna(0).to_numpy())
                input_df[cum_col_name] = [st.query(i) for i in range(len(input_df))]

    # 3. Compute Autoencoder reconstruction error
    if "reconstruction_error" in FEATURES_XGB and mlp_ae:
        available_cols = [c for c in continuous_cols if c in input_df.columns]
        print(f"available cols "+str(available_cols))

        if available_cols:
            input_array = input_df[available_cols].to_numpy().astype(float)
            
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            
            predictions = mlp_ae.predict(input_array)
            
            if predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)
            
            reconstruction_error = np.mean((input_array - predictions)**2, axis=1)
            input_df["reconstruction_error"] = reconstruction_error
        else:
            input_df["reconstruction_error"] = 0

    # 🔑 Garante que TODAS as colunas esperadas estejam presentes e na ordem correta.
    # 🔑 reindex() remove automaticamente colunas extras (like 'koi_tce_plnt_num')
    # 🔑 e adiciona as faltantes (preenchendo com fill_value=0).
    input_df = input_df.reindex(columns=FEATURES_XGB, fill_value=0)

    return input_df


# Metrics Calculation Function
def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics including accuracy, F1-score,
    confusion matrix, and a full classification report.
    """
    # Check if y_true (y_test) contains any data
    if y_true.empty:
         return {"note": "Cannot compute metrics: y_test data is not available."}

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        # Convert NumPy array to list for JSON serialization
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics


# Flask Endpoint for JSON Input
@app.route('/predict_json', methods=['POST'])
def predict_json():
    if not bst_ae:
        return jsonify({"error": "ML Model not loaded on server."}), 500

    data = request.get_json()
    print("OLHA")
    print
    # Convert single dict to DataFrame if needed
    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    elif isinstance(data, list):  # If list of dicts
        input_df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Invalid JSON format. Expected object or list of objects."}), 400

    # Preprocess input
    processed = preprocess_input(input_df)

    # Predict with XGBoost
    dmat = xgb.DMatrix(processed.values, feature_names=FEATURES_XGB)
    preds = bst_ae.predict(dmat).astype(int)

    # Compute metrics only if input length matches y_test
    if len(input_df) == len(y_test) and not y_test.empty:
        try:
            metrics = compute_metrics(y_test, preds)
        except Exception as e:
            print(f"Error computing metrics: {e}")
            metrics = {"note": "Error calculating metrics. Check if y_test is correctly mapped to prediction labels."}
    else:
        metrics = {"note": f"Metrics are calculated only if input length ({len(input_df)}) matches y_test length ({len(y_test)})."}

    # Return predictions and metrics
    return jsonify({
        "predictions": preds.tolist(),
        "metrics": metrics
    })


# Flask Endpoint for CSV Upload
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    print("oi mundo")
    if not bst_ae:
        return jsonify({"error": "ML Model not loaded on server."}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo CSV foi enviado."}), 400
        
        file = request.files['file']

   
        try:
            # Read file content into memory
            file_content = file.read()
            # Try reading with automatic separator detection
            input_df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep=None, engine='python', comment='#')
        except Exception:
            # Fallback to standard comma separator
            input_df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep=',', comment='#', engine='python')

        print(">>> CSV lido com sucesso!")
        print(">>> COLUNAS LIDAS:", input_df.columns.tolist())
        print("AAAAAAAAAAAAAAAAAAAAA")
        print(">>> DIMENSÕES:", input_df.shape)

        # ✅ Pré-processar dados
        processed = preprocess_input(input_df)

        print(">>> COLUNAS FINAL:", len(processed.columns))
        print(">>>- SHAPE FINAL:", processed.shape)
       

        dmat = xgb.DMatrix(processed.values, feature_names=FEATURES_XGB)
        
        preds = bst_ae.predict(dmat).astype(int)
        print("Previsoes")
        print(preds)
        if len(input_df) == len(y_test) and not y_test.empty:
            metrics = compute_metrics(y_test, preds)
        else:
            metrics = {"note": f"Métricas calculadas apenas se o tamanho do input ({len(input_df)}) for igual ao conjunto de teste ({len(y_test)})."}

        return jsonify({
            "predictions": preds.tolist(),
            "metrics": metrics
        })

    except Exception as e:
        print("ERRO AO PROCESSAR CSV:", e)
        return jsonify({"error": str(e)}), 400

@app.route('/get-planets', methods=['GET'])
def get_planetas():
    if planets.empty:
        return jsonify({"error": "Kepler data not loaded on server."}), 500

    confirmed_planets = planets[planets['koi_disposition'] == 'CONFIRMED']
    
    eight_planets = confirmed_planets.iloc[:8]
    
    # Converte valores para tipos nativos (float, int, str)
    lista_json = eight_planets.astype(object).where(pd.notnull(eight_planets), None).to_dict(orient="records")
    
    return jsonify(lista_json)

# Run Flask App
if __name__ == '__main__':
    # Setting use_reloader=False prevents the app from running twice if debug=True
    app.run(port=5090, debug=True, use_reloader=False)
