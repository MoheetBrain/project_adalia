"""
ADALIA STATE 5: Train Muguet Odor Classifier
Learns structure-odor relationship from curated data
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🧠 ADALIA: Training Muguet Odor Prediction Model")
print("=" * 70)

# STEP 1: Load training data
print("\n📚 Loading training data...")
train_df = pd.read_csv('expanded_odor_training.csv')
print(f"   ✅ {len(train_df)} training molecules loaded")
print(f"   📊 Muguet molecules: {train_df['is_muguet'].sum()}")
print(f"   📊 Non-muguet molecules: {(1-train_df['is_muguet']).sum()}")

# STEP 2: Compute molecular fingerprints (this encodes structure)
print("\n🔬 Computing molecular fingerprints...")

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint (ML-ready vector)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

# Compute fingerprints for training set
X_train = np.array([smiles_to_fingerprint(s) for s in train_df['smiles']])
y_train = train_df['is_muguet'].values

print(f"   ✅ Fingerprint matrix: {X_train.shape}")

# STEP 3: Train the classifier
print("\n🎯 Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # Handle imbalanced muguet/non-muguet
)

model.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
print(f"   ✅ Model trained!")
print(f"   📊 Cross-val AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# STEP 4: Load ADALIA candidates and predict
print("\n🔍 Scoring ADALIA candidates...")
adalia_df = pd.read_csv('ligands.csv')

# Compute fingerprints for candidates
X_adalia = np.array([smiles_to_fingerprint(s) for s in adalia_df['smiles']])

# Predict muguet probability
muguet_probs = model.predict_proba(X_adalia)[:, 1]  # Probability of class=1 (muguet)
adalia_df['Muguet_Probability'] = np.round(muguet_probs, 3)

# STEP 5: Merge with existing analysis and re-rank
print("\n📊 Merging with existing property data...")
old_analysis = pd.read_csv('adalia_ranked.csv')

# Add muguet probability
old_analysis['Muguet_Probability'] = adalia_df['Muguet_Probability']

# NEW COMPOSITE SCORE (includes ML prediction)
old_analysis['ML_Score'] = (
    old_analysis['Muguet_Probability'] * 0.4 +      # ML odor prediction (40%)
    old_analysis['Lilial_Similarity'] * 0.3 +       # Fingerprint similarity (30%)
    (1 - old_analysis['Has_Aldehyde']) * 0.2 +      # Safety (20%)
    (1 - abs(old_analysis['LogP'] - 3.5) / 5).clip(0, 1) * 0.1  # Volatility (10%)
)

# Re-sort
old_analysis = old_analysis.sort_values('ML_Score', ascending=False)

# Save
old_analysis.to_csv('adalia_ml_ranked.csv', index=False)

print("\n" + "=" * 70)
print("🏆 TOP 5 CANDIDATES (ML-Enhanced Ranking)")
print("=" * 70)

display_cols = ['name', 'family', 'Muguet_Probability', 'Lilial_Similarity', 
                'LogP', 'Has_Aldehyde', 'ML_Score']
print(old_analysis[display_cols].head(5).to_string(index=False))

print("\n" + "=" * 70)
print("✅ ML-ranked results saved: 'adalia_ml_ranked.csv'")
print
 