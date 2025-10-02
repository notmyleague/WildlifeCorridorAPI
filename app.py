import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- 1. SETUP AND DATA LOADING ---
# NOTE: This assumes you have a single CSV file containing all features and the target.
# This file must be the output of the data prep phase (Mr. Shivansh, Mr. Govind, etc.)

DATA_FILE = 'master_feature_dataset.csv'
TARGET_COLUMN = 'species_presence' # Must be 1 (Presence) or 0 (Absence)
MODEL_OUTPUT_FILE = 'xgboost_hsm_model.json'

try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Data file '{DATA_FILE}' not found. Please ensure the path is correct.")
    exit()

# Identify features (X) and target (Y)
# Exclude non-feature columns like 'latitude', 'longitude', and the target column itself
EXCLUDE_COLS = ['latitude', 'longitude', TARGET_COLUMN]
FEATURES = [col for col in df.columns if col not in EXCLUDE_COLS]

X = df[FEATURES]
y = df[TARGET_COLUMN]

# --- 2. TRAIN/TEST SPLIT (Done by Mr. Rishabh Tripathii, but needed for training) ---
# We split the data to train the model and generate metrics (for Mr. Akarsh)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# --- 3. APPLY XGBOOST CLASSIFIER ---
# Initialize the model:
# objective='binary:logistic' makes the model output a probability (0 to 1)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,        # Number of boosting rounds (trees)
    learning_rate=0.05,      # Step size shrinkage
    max_depth=5,             # Maximum tree depth
    use_label_encoder=False, # Suppress warning for new XGBoost versions
    eval_metric='logloss',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print("\nXGBoost Model Training Complete.")

# --- 4. PREDICTION AND INITIAL METRICS (To be formally analyzed by Mr. Akarsh) ---

# Predict probabilities (Habitat Suitability Score) on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int) # Use 0.5 as default classification threshold

# Calculate AUC Score (A key metric for Habitat Suitability Models)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- Initial Model Performance (for verification) ---")
print(f"Test Set AUC Score (Habitat Suitability): {auc_score:.4f}")
print("\nClassification Report (Threshold 0.5):")
print(classification_report(y_test, y_pred_class))

# --- 5. SAVE THE TRAINED MODEL ---
# Save the model so it can be used for prediction on the full raster grid later.
model.save_model(MODEL_OUTPUT_FILE)
print(f"\nModel saved successfully to: {MODEL_OUTPUT_FILE}")

# --- 6. NEXT STEP: GENERATE PROBABILITY MAP (Conceptual for now) ---
# The next critical step is to load the model and apply it to a grid of all pixels.
# This part is placeholder code to illustrate the final Habitat Suitability Map generation:

# # Placeholder: Assuming you have a full raster features grid 'X_full_grid'
# # X_full_grid = rasterio.open('full_feature_stack.tif').read().reshape(...)
# # full_suitability_map = model.predict_proba(X_full_grid)[:, 1]
# # Save 'full_suitability_map' as a new GeoTIFF.