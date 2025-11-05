# remove this – it's unused and can confuse static analyzers
# from urllib.request import DataHandler

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import *


class DataSetHandler:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.validation_split = (1.0 - DATA_SPLIT)
        self.fault_columns = FAULT_COLUMNS

        # ensure plot dir exists
        self.plot_dir = "plots"
        os.makedirs(self.plot_dir, exist_ok=True)

        # --- Load dataset ---
        self.table = pd.read_csv(INPUT_FILE)
        print(f"✅ Loaded dataset with shape: {self.table.shape}")

        # --- Separate input/output ---
        # Make outputs explicitly numeric so they survive correlation
        self.output_df = self.table[[c for c in self.fault_columns if c in self.table.columns]].apply(
            pd.to_numeric, errors="coerce"
        )

        # Inputs = everything else
        self.input_df = self.table.drop(
            columns=[c for c in self.fault_columns if c in self.table.columns],
            errors="ignore"
        )
        if 'target' in self.input_df.columns:
            self.input_df = self.input_df.drop(columns='target')

        self.reduced_inputs = None
        self.normalized_inputs = None
        self.df_clean = None
        self.X_train = self.X_val = self.Y_train = self.Y_val = None

    # ----------------------------------------------------------------
    def show_input_heatmap(self, save_path=None):
        corr = self.input_df.corr(numeric_only=True).abs()
        if corr.empty:
            print("⚠️ No numeric input columns to correlate.")
            return
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.3)
        plt.title("Correlation Heatmap (Inputs – Before Normalization)", fontsize=14, pad=20)
        plt.tight_layout()
        save_path = save_path or os.path.join(self.plot_dir, "heatmap_inputs.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"🖼️ Saved input heatmap → {save_path}")

    # ----------------------------------------------------------------
    def show_input_output_correlation(self, save_path=None):
        """Save correlation heatmap between inputs and outputs (before normalization)."""
        # Combine and coerce to numeric; keep columns even if they have NaNs
        combined = pd.concat([self.input_df, self.output_df], axis=1).apply(pd.to_numeric, errors='coerce')
        corr = combined.corr()

        # only keep columns that actually exist in corr
        input_cols  = [c for c in self.input_df.columns  if c in corr.columns]
        output_cols = [c for c in self.output_df.columns if c in corr.columns]

        if not input_cols:
            print("⚠️ No numeric input columns found for input/output correlation.")
            return
        if not output_cols:
            print("⚠️ No numeric output columns found for input/output correlation.")
            return

        sub_corr = corr.loc[input_cols, output_cols]
        if sub_corr.empty:
            print("⚠️ Correlation matrix between inputs and outputs is empty (likely due to NaNs or constant columns).")
            return

        plt.figure(figsize=(max(8, len(output_cols) * 1.2), max(6, len(input_cols) * 0.4)))
        sns.heatmap(sub_corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
        plt.title("Correlation: Inputs vs Outputs (Before Normalization)", fontsize=14, pad=20)
        plt.tight_layout()

        save_path = save_path or os.path.join(self.plot_dir, "heatmap_inputs_vs_outputs.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"🖼️ Saved input–output heatmap → {save_path}")





handler = DataSetHandler()
handler.show_input_heatmap()
handler.show_input_output_correlation()