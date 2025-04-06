# Required packages: pip install scikit-learn pandas numpy torch
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os

def get_data_path():
    """
    Returns the path to the data file, working in both scripts and Jupyter notebooks.
    Assumes the data file is in the same directory as the script/notebook.
    """
    try:
        # For regular Python scripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, 'spambase.data')
    except NameError:
        # For Jupyter notebooks
        return 'spambase.data'

# Load the data
# Note: The last column is the target variable (spam = 1, non-spam = 0)
try:
    data = pd.read_csv(get_data_path(), header=None)
except FileNotFoundError:
    print("Error: Could not find 'spambase.data'")
    print("Please ensure 'spambase.data' is in the same directory as this script/notebook")
    raise

# Split into features and target
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column (target)

# Scale the features for SVM and GRU
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to PyTorch tensors for GRU
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y.values)

# Custom Dataset for GRU
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# GRU Model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Reshape input for GRU: [batch_size, sequence_length, input_size]
        x = x.unsqueeze(1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out.squeeze()

def evaluate_gru_model(X, y, model_name):
    """
    Evaluates GRU model using 10-fold stratified cross-validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create data loaders
        train_dataset = SpamDataset(X_train, y_train)
        test_dataset = SpamDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model
        model = GRUClassifier(input_size=X.shape[1], hidden_size=64).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        model.train()
        for epoch in range(10):  # 10 epochs per fold
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions = (outputs > 0.5).float().cpu().numpy()
                y_pred.extend(predictions)
                y_true.extend(batch_y.numpy())
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        tss = tpr + tnr - 1
        hss = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0

        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fdr = fp / (fp + tp) if (fp+tp) > 0 else 0
        
        fold_metrics.append({
            'Fold': fold + 1,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr,
            'TSS': tss, 'HSS': hss,
            'Prec.': precision, 
            'NPV': npv,    
            'F1': f1,         
            'Sens.': tpr,  
            'Specif.': tnr,   
            'Acc.': accuracy,
            'FDR': fdr,
            'Confusion Matrix': [[tp, fn], [fp, tn]]  
        })
        

    fold_metrics_df = pd.DataFrame(fold_metrics)
    average_metrics = fold_metrics_df.mean(numeric_only=True).to_dict()
    average_metrics['Fold'] = 'Average'
    average_metrics_df = pd.DataFrame([average_metrics])
    results_df = pd.concat([fold_metrics_df, average_metrics_df], ignore_index=True)
    return results_df


# Load the data
# Note: The last column is the target variable (spam = 1, non-spam = 0)
data = pd.read_csv('c:/Users/clagg/Downloads/spambase/spambase.data', header=None)

# Split into features and target
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column (target)

# Scale the features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def evaluate_model(model, X, y, model_name):
    """
    Evaluates a given model using 10-fold stratified cross-validation and calculates
    performance metrics.

    Args:
        model: The machine learning model to evaluate (e.g., RandomForestClassifier, SVC).
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

        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        fold_metrics.append({
            'Fold': fold + 1,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr,
            'TSS': tss, 'HSS': hss,
            'Prec.': precision,
            'NPV': npv,       
            'F1': f1,           
            'Sens.': tpr,  
            'Specif.': tnr,   
            'Acc.': accuracy,     
            'FDR': fdr,
            'Confusion Matrix': [[tp, fn], [fp, tn]]
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
    'Sens.': '{:.4f}'.format,
    'Specif.': '{:.4f}'.format,
    'Prec.': '{:.4f}'.format,
    'NPV': '{:.4f}'.format,
    'FPR': '{:.4f}'.format,
    'FDR': '{:.4f}'.format,
    'FNR': '{:.4f}'.format,
    'Acc.': '{:.4f}'.format,
    'F1': '{:.4f}'.format,
    'TSS': '{:.4f}'.format,
    'HSS': '{:.4f}'.format
}))

# 2. SVM Implementation
print("\nSupport Vector Machine Results:")
# Using RBF kernel with optimized parameters
svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_results = evaluate_model(svm_classifier, X_scaled, y, "SVM")
print(svm_results.to_string(index=False, formatters={
    'Sens.': '{:.4f}'.format,
    'Specif.': '{:.4f}'.format,
    'Prec.': '{:.4f}'.format,
    'NPV': '{:.4f}'.format,
    'FPR': '{:.4f}'.format,
    'FDR': '{:.4f}'.format,
    'FNR': '{:.4f}'.format,
    'Acc.': '{:.4f}'.format,
    'F1': '{:.4f}'.format,
    'TSS': '{:.4f}'.format,
    'HSS': '{:.4f}'.format
}))

# 3 GRU evaluation
print("\nGRU Results:")
gru_results = evaluate_gru_model(X_tensor, y_tensor, "GRU")
print(gru_results.to_string(index=False, formatters={
    'Sens.': '{:.4f}'.format,
    'Specif.': '{:.4f}'.format,
    'Prec.': '{:.4f}'.format,
    'NPV': '{:.4f}'.format,
    'FPR': '{:.4f}'.format,
    'FDR': '{:.4f}'.format,
    'FNR': '{:.4f}'.format,
    'Acc.': '{:.4f}'.format,
    'F1': '{:.4f}'.format,
    'TSS': '{:.4f}'.format,
    'HSS': '{:.4f}'.format
}))

# Show feature importance for Random Forest
rf_classifier.fit(X, y)
feature_importances = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importances.head(10).to_string(index=False))