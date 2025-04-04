# Required packages: pip install scikit-learn pandas numpy tensorflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the data
# Note: The last column is the target variable (spam = 1, non-spam = 0)
data = pd.read_csv('c:/Users/clagg/Downloads/spambase/spambase.data', header=None)

# Split into features and target
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column (target)

# Scale the features for SVM and GRU
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for GRU (samples, timesteps, features)
# We'll treat each feature as a timestep
X_gru = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

def create_gru_model(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def evaluate_model(model, X, y, model_name):
    """
    Evaluates a given model using 10-fold stratified cross-validation and calculates
    performance metrics.

    Args:
        model: The machine learning model to evaluate (e.g., RandomForestClassifier, SVC, or GRU model).
        X: The feature data.
        y: The target data.
        model_name (str):  The name of the model
    Returns:
        pandas.DataFrame: A DataFrame containing the performance metrics for each fold
                          and the average metrics.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Convert indices to numpy arrays if they aren't already
        train_index = np.array(train_index)
        test_index = np.array(test_index)
        
        # Handle both numpy arrays and pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
            
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == "GRU":
            # For GRU, we need to reshape the data
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            # Create and train a new model for each fold
            fold_model = create_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            fold_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred = (fold_model.predict(X_test) > 0.5).astype(int)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        tss = tpr + tnr - 1
        hss = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0

        fold_metrics.append({
            'Fold': fold + 1,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr,
            'TSS': tss, 'HSS': hss
        })

    fold_metrics_df = pd.DataFrame(fold_metrics)
    average_metrics = fold_metrics_df.mean(numeric_only=True).to_dict()
    average_metrics['Fold'] = 'Average'
    average_metrics_df = pd.DataFrame([average_metrics])
    results_df = pd.concat([fold_metrics_df, average_metrics_df], ignore_index=True)
    return results_df

# 1. Random Forest Implementation
print("Random Forest Results:")
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_results = evaluate_model(rf_classifier, X, y, "Random Forest")
print(rf_results.to_string(index=False, formatters={
    'TPR': '{:.4f}'.format, 'TNR': '{:.4f}'.format,
    'FPR': '{:.4f}'.format, 'FNR': '{:.4f}'.format,
    'TSS': '{:.4f}'.format, 'HSS': '{:.4f}'.format
}))

# 2. SVM Implementation
print("\nSupport Vector Machine Results:")
# Using RBF kernel with optimized parameters
svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_results = evaluate_model(svm_classifier, X_scaled, y, "SVM")
print(svm_results.to_string(index=False, formatters={
    'TPR': '{:.4f}'.format, 'TNR': '{:.4f}'.format,
    'FPR': '{:.4f}'.format, 'FNR': '{:.4f}'.format,
    'TSS': '{:.4f}'.format, 'HSS': '{:.4f}'.format
}))

# 3. GRU Implementation
print("\nGRU Results:")
gru_model = create_gru_model(input_shape=(1, X_scaled.shape[1]))
gru_results = evaluate_model(gru_model, X_gru, y, "GRU")
print(gru_results.to_string(index=False, formatters={
    'TPR': '{:.4f}'.format, 'TNR': '{:.4f}'.format,
    'FPR': '{:.4f}'.format, 'FNR': '{:.4f}'.format,
    'TSS': '{:.4f}'.format, 'HSS': '{:.4f}'.format
}))

# Show feature importance for Random Forest
rf_classifier.fit(X, y)
feature_importances = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importances.head(10).to_string(index=False))