import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


# 1. Load dataset
input_file = "../faults.csv"
df = pd.read_csv(input_file)
print(f"✅ Loaded dataset with shape: {df.shape}")


# 2. Separate input (X) and output (Y)
fault_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
output_df = df[[col for col in fault_columns if col in df.columns]]

input_df = df.drop(columns=[col for col in fault_columns if col in df.columns], errors="ignore")
if 'target' in input_df.columns:
    input_df = input_df.drop(columns='target')

# 3. Correlation + heatmap
corr = input_df.corr(numeric_only=True).abs()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr, cmap="coolwarm", center=0,
    annot=False, square=True, linewidths=0.3,
    cbar_kws={"shrink": 0.8}
)
plt.title("Test Heatmap", fontsize=14, pad=20)
plt.tight_layout()
plt.show()


# 4. Find redundant features

threshold = 0.95
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

print("\n📊 Highly correlated feature pairs (>|0.95|):")
for col in upper.columns:
    for row in upper.index:
        if upper.loc[row, col] > threshold:
            print(f"  {row:25s} ↔ {col:25s} = {upper.loc[row, col]:.3f}")

print("\n🧹 Suggested features to drop (keeping one from each correlated group):")
print(to_drop if to_drop else "None found — dataset already clean!")

# 5. Drop redundant features
reduced_inputs = input_df.drop(columns=to_drop, errors='ignore')

# 6. Normalize the input data
scaler = StandardScaler()
normalized_inputs = pd.DataFrame(
    scaler.fit_transform(reduced_inputs),
    columns=reduced_inputs.columns
)

print("\n📈 Normalization complete. Mean ~0, Std ~1 for each feature.")

# 7. Combine normalized inputs + outputs
df_clean = pd.concat([normalized_inputs, output_df.reset_index(drop=True)], axis=1)

output_file = "faults_clean_normalized.csv"
df_clean.to_csv(output_file, index=False)

