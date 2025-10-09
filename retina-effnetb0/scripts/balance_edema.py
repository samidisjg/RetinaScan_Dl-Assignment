import pandas as pd, os, sys
inp = "data_preprocessed/train/annotations.csv"
out = "data_preprocessed/train/annotations_balanced.csv"
df = pd.read_csv(inp)
pos = df[df["Risk of macular edema"]==1]
neg = df[df["Risk of macular edema"]==0]
# duplicate positives until ~balance (tweak factor if needed)
factor = max(1, int(len(neg)/max(1,len(pos))) - 1)
df_bal = pd.concat([df, *([pos]*factor)], ignore_index=True).sample(frac=1.0, random_state=42)
df_bal.to_csv(out, index=False)
print("Wrote:", out, "shape:", df_bal.shape)
