
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Λειτουργία καθαρισμού δεδομένων
def clean_data(df):
   # καθαριζουμε τα δεδομενα  αντικαθιστώντας μη αριθμητικές τιμές με NaN και στη συνέχεια αντικαθιστά τα NaN με τη μέση τιμή κάθε στήλης
    df = df.apply(pd.to_numeric, errors='coerce')  # Μετατροπή σε αριθμούς
    df = df.fillna(df.mean())  # Αντικατάσταση NaN με τη μέση τιμή κάθε στήλης
    return df

# Φόρτωση δεδομένων
data = pd.read_csv("training_companydata.csv")
test_data = pd.read_csv("test_unlabeled.csv", header=None)

# Αντιστοίχιση σωστών ονομάτων στις στήλες του test_data
test_data.columns = data.columns[:-1]

# Διαχωρισμός χαρακτηριστικών και κλάσης στόχου
X = clean_data(data.iloc[:, :-1])  # Καθαρισμός χαρακτηριστικών
y = data.iloc[:, -1]  # X65: Κλάση στόχος
test_data = clean_data(test_data)  # Καθαρισμός δεδομένων test

# Διαχωρισμός συνόλου για εκπαίδευση και επικύρωση
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversampling με SMOTE για αντιμετώπιση ανισορροπίας
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Κλιμάκωση δεδομένων
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_val = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data)

# Εκπαίδευση μοντέλου Gradient Boosting με βελτιώσεις
gb_model = GradientBoostingClassifier(
    random_state=42,
    n_estimators=1000,         # Λιγότερα δέντρα για αποφυγή υπερπροσαρμογής
    max_depth=4,              # Μικρότερο βάθος δέντρων
    learning_rate=0.01,       # Μικρό learning rate για σταθερή εκπαίδευση στο μοντέλο μας
    subsample=0.9,            # Υποδειγματοληψία για αποφυγή υπερπροσαρμογής
    min_samples_split=100,    # Αύξηση ελάχιστου αριθμού δειγμάτων για διάσπαση
    min_samples_leaf=50,     # Αύξηση ελάχιστων δειγμάτων ανά φύλλο
    max_features="sqrt"      # Χρήση τυχαίου υποσυνόλου χαρακτηριστικών
)
gb_model.fit(X_train_resampled, y_train_resampled)

# Αξιολόγηση στο validation set (Με όλα τα γνωρισματα)
y_val_pred_gb = gb_model.predict(X_val)
print("\nClassification Report - Validation Set (Improved Model)")
print(classification_report(y_val, y_val_pred_gb))

# Αξιολόγηση στο training set για ανίχνευση overfitting
y_train_pred_gb = gb_model.predict(X_train_resampled)
print("Classification Report - Training Set")
print(classification_report(y_train_resampled, y_train_pred_gb))

# Καμπύλη Εκμάθησης για λλογους debugging
train_sizes, train_scores, val_scores = learning_curve(
    gb_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
)

# Μέση και τυπική απόκλιση των σκορ
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Σχεδίαση γραφημάτων για λογουσ debugging
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training score", color="blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.plot(train_sizes, val_mean, label="Validation score", color="green")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Διασταυρούμενη επικύρωση για αξιόπιστη εκτίμηση απόδοσης
cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring="accuracy")
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# Προβλέψεις για το test set (Βελτιωμένο Μοντέλο)
predictions_full = gb_model.predict(test_data_scaled)

# Αποθήκευση προβλέψεων στο αρχείο (Με ολα τα γνωρισματα)
predictions_df_full = pd.DataFrame(predictions_full, columns=["Prediction"])
predictions_df_full.to_csv("predictions.csv", index=False) #πρωτο ζητουμενο αρχειο

print("\nΤο αρχείο 'predictions.csv' δημιουργήθηκε με προβλέψεις από το  μοντέλο με όλα τα γνωρισμάτα")
probabilities_full = gb_model.predict_proba(test_data_scaled)[:, 1]

# Δημιουργία DataFrame για τις πιθανότητες πτώχευσης
results_df = pd.DataFrame({
    "RowID": range(1, len(test_data) + 1),  # RowID ξεκινά από 1 όπωσ ζητηθηκε
    "Risk_Probability": probabilities_full
})

# Ταξινόμηση κατά φθίνουσα πιθανότητα και επιλογή των 50 πρώτων
top_50 = results_df.sort_values(by="Risk_Probability", ascending=False).head(50)

# Αποθήκευση των αποτελεσμάτων στο αρχείο
top_50.to_csv("top_50_high_risk.csv", index=False) #δευτερο ζητουμενο αρχειο

print("\nΤο αρχείο 'top_50_high_risk.csv' δημιουργήθηκε με τις 50 εταιρείες που φαίνεται πιθανότερο να χρεοκοπήσουν.")



#  Εύρεση 10 κορυφαίων χαρακτηριστικών
feature_importances = gb_model.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]
top_features = features[sorted_idx][:10]
print("\nTop 10 features:", top_features)

# Δημιουργία DataFrame με μόνο τα κορυφαία χαρακτηριστικά
X_top = X[top_features]
X_train_top, X_val_top, y_train_top, y_val_top = train_test_split(
    X_top, y, test_size=0.2, random_state=42, stratify=y
)

# Oversampling για το νέο σύνολο δεδομένων
X_train_resampled_top, y_train_resampled_top = smote.fit_resample(X_train_top, y_train_top)

# Κλιμάκωση δεδομένων για τα κορυφαία χαρακτηριστικά
X_train_resampled_top = scaler.fit_transform(X_train_resampled_top)
X_val_top = scaler.transform(X_val_top)

# Εκπαίδευση μοντέλου ΜΟΝΟ με τα κορυφαία χαρακτηριστικά
gb_model_top = GradientBoostingClassifier(
   random_state=42,
    n_estimators=1000,         # Λιγότερα δέντρα για αποφυγή υπερπροσαρμογής
    max_depth=4,              # Μικρότερο βάθος δέντρων
    learning_rate=0.01,       # Μικρό learning rate για σταθερή εκπαίδευση
    subsample=0.9,            # Υποδειγματοληψία για αποφυγή υπερπροσαρμογής
    min_samples_split=100,    # Αύξηση ελάχιστου αριθμού δειγμάτων για διάσπαση
    min_samples_leaf=50,  
)
gb_model_top.fit(X_train_resampled_top, y_train_resampled_top)

# Αξιολόγηση στο validation set για το μοντέλο με τα κορυφαία χαρακτηριστικά
y_val_pred_gb_top = gb_model_top.predict(X_val_top)
print("\nClassification Report - Validation Set (Top Features)")
print(classification_report(y_val_top, y_val_pred_gb_top))
