import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# 1. LOAD & NORMALIZE
# ==========================================
df = pd.read_csv('fixed_data_clean.csv')
meta_cols = ['CONS_NO', 'FLAG']
date_cols = [c for c in df.columns if c not in meta_cols]

# Perform Row-wise Max Normalization (Standardize each user to peak of 1.0)
# We assume NAs are already handled as per your instructions.
consumption = df[date_cols]
row_max = consumption.max(axis=1).replace(0, 1) # Avoid division by zero
norm_cons = consumption.div(row_max, axis=0)

# ==========================================
# 2. FEATURE ENGINEERING (Behavioral)
# ==========================================
def get_behavioral_features(raw_df, normalized_df):
    X = pd.DataFrame(index=raw_df.index)
    # Filter Heuristic: Count actual zeros
    X['zero_pct'] = (normalized_df == 0).sum(axis=1) / len(date_cols)
    # Trend Heuristic: Recent vs Past consumption ratio
    split = int(len(date_cols) * 0.8)
    past = normalized_df.iloc[:, :split].mean(axis=1)
    recent = normalized_df.iloc[:, split:].mean(axis=1)
    X['drop_ratio'] = (recent + 0.01) / (past + 0.01)
    # Variability
    X['volatility'] = normalized_df.std(axis=1) / (normalized_df.mean(axis=1) + 0.01)
    return X

features = get_behavioral_features(df, norm_cons)

# ==========================================
# 3. 4-PASS ACTIVE LEARNING LOOP
# ==========================================
BATCH_SIZE = 100
uninspected_indices = list(df.index)
inspected_indices = []
pass_history = []

for p in range(1, 10):
    if p == 1:
        # Pass 1: Ignore zero-heavy users initially (Heuristic Filter)
        active_pool = [i for i in uninspected_indices if features.loc[i, 'zero_pct'] < 0.4]
        target_pool = active_pool if len(active_pool) >= BATCH_SIZE else uninspected_indices
        
        iso = IsolationForest(random_state=42).fit(features.loc[target_pool])
        scores = iso.decision_function(features.loc[target_pool])
        candidates = pd.Series(scores, index=target_pool).sort_values().head(BATCH_SIZE).index.tolist()
    else:
        # Passes 2-4: Supervised learning on revealed labels
        X_train = features.loc[inspected_indices]
        y_train = df.loc[inspected_indices, 'FLAG']
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        probs = clf.predict_proba(features.loc[uninspected_indices])[:, 1]
        candidates = pd.Series(probs, index=uninspected_indices).sort_values(ascending=False).head(BATCH_SIZE).index.tolist()

    inspected_indices.extend(candidates)
    uninspected_indices = [i for i in uninspected_indices if i not in candidates]
    pass_history.append({'Pass': p, 'Thieves_Found': df.loc[candidates, 'FLAG'].sum()})

# ==========================================
# 4. CLUSTERING THIEF ARCHETYPES
# ==========================================
# Isolate confirmed thieves from our inspections
thieves_found = df.loc[inspected_indices]
thieves_found = thieves_found[thieves_found['FLAG'] == 1]

# Create 3 "Theft Archetypes" based on normalized shape similarity
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(norm_cons.loc[thieves_found.index])

# Add "Distance to Archetype" as a supervised feature for all users
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    features[f'dist_to_thief_type_{i}'] = np.linalg.norm(norm_cons.values - center, axis=1)

# ==========================================
# 5. FINAL ESTIMATION & STATISTICS
# ==========================================
# Train the master model using all inspections + cluster features
final_model = RandomForestClassifier(n_estimators=300, random_state=42)
final_model.fit(features.loc[inspected_indices], df.loc[inspected_indices, 'FLAG'])

# Apply to the entire 42k population
df['final_prob'] = final_model.predict_proba(features)[:, 1]
threshold = df['final_prob'].quantile(0.995) # Flag top 8%
df['final_pred'] = (df['final_prob'] >= threshold).astype(int)

# --- Output Report ---
y_true = df['FLAG']
y_pred = df['final_pred']
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

print("\n" + "="*40)
print("   ITERATIVE DISCOVERY RESULTS")
print("="*40)
print(pd.DataFrame(pass_history))

print("\n" + "="*40)
print("   FINAL SYSTEM STATISTICS")
print("="*40)
print(f"Total Population: {len(df)}")
print(f"True Positives:  {tp} (Thieves Correctly Flagged)")
print(f"False Positives: {fp} (Honest Users Misidentified)")
print(f"Precision:       {tp/(tp+fp):.2%}")
print(f"Recall:          {tp/(tp+fn):.2%}")
print(f"Accuracy:        {accuracy_score(y_true, y_pred):.2%}")
print("="*40)

# --- Visualization: Normalized Correlation Heatmap ---
plt.figure(figsize=(10, 8))
sample_idx = list(df[df['FLAG']==1].head(15).index) + list(df[df['FLAG']==0].head(15).index)
corr_matrix = norm_cons.loc[sample_idx].T.corr()
sns.heatmap(corr_matrix, cmap='RdYlBu_r', center=0)
plt.title("Correlation Heatmap: Normalized Shape Similarity")
plt.show()
