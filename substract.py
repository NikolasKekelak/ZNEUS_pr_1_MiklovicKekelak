import pandas as pd

# === CONFIG ===
input_file = "faults.csv"                # original file
output_file = "faults.csv"  # output file
target_column = "Class"                  # column to modify

# === LOAD ===
df = pd.read_csv(input_file)
print(f"✅ Loaded dataset with shape: {df.shape}")

# === CHECK + MODIFY ===
if target_column not in df.columns:
    print(f"⚠️ Column '{target_column}' not found in dataset.")
else:
    # Subtract 1 from every value in the column
    df[target_column] = df[target_column] - 1
    print(f"🧮 Subtracted 1 from every value in '{target_column}'.")

# === SAVE ===
df.to_csv(output_file, index=False)
print(f"💾 Saved updated dataset → {output_file}")