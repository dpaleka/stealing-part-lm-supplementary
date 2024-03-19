# %%
"""
1. Load the arrays from the .pkl files
2. Determine all values which are always (or almost always, say more than 50% of the time) NaN
3. Impute NaNs using averages of non-NaN values in the same array
4. Save the imputed matrix to a .pkl file
5. Run SVD on the imputed matrix and return plot of singular values
"""
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tqdm import tqdm

model_name = "EleutherAI/pythia-70m"
RESPONSE_DIR = f"logits/{model_name}_qsimple_a"


# %%
# 1. Load the arrays from the .pkl files
data = []
for pkl_file in tqdm(list(Path(RESPONSE_DIR).glob("*.pkl"))):
    with open(pkl_file, "rb") as f:
        data.append(pickle.load(f))

# Make matrix
data_matrix = np.vstack(data)
print(f"{data_matrix.shape=}")
N, vocab_size = data_matrix.shape

# %%

# 2. Determine all values which are always (or almost always, say more than 50% of the time) NaN
nan_counts = np.isnan(data_matrix).sum(axis=0)
threshold = N * 0.5
columns_to_drop = np.where(nan_counts > threshold)[0]


# %%
print(f"Number of columns to drop: {len(columns_to_drop)}")
print(f"Number of columns to keep: {vocab_size - len(columns_to_drop)}")
print(f"{data_matrix.shape=}")

# %%
# Drop columns where more than 50% of the values are NaN
data_matrix_semiclean = np.delete(data_matrix, columns_to_drop, axis=1)

# %%
print(f"{data_matrix_semiclean.shape=}")

# %%
# 3. Impute NaNs using averages of non-NaN values in the same array
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
data_imputed = imputer.fit_transform(data_matrix_semiclean)

# %%
# Sample first m=1000 rows
m = 1000
data_imputed_sample = data_imputed[:m, :]

# %%
# 4. Save the imputed matrix to a .pkl file
SAVE_DIR = Path("imputed_matrices")
imputed_filename = SAVE_DIR / f"{model_name}_imputed_matrix.pkl"
imputed_filename.parent.mkdir(exist_ok=True, parents=True)
with open(imputed_filename, "wb") as f:
    pickle.dump(data_imputed_sample, f)

# %%
# 5. Run SVD on the imputed matrix and return plot of singular values
# svd = TruncatedSVD(n_components=min(data_imputed.shape) - 1)
# svd.fit(data_imputed)
# singular_values = svd.singular_values_

# use numpy svd
u, s, vh = np.linalg.svd(data_imputed_sample, full_matrices=False)
singular_values = s

# %%
eps = 1e-2
# Find the number of singular values which are greater than eps
n_singular_values = np.sum(singular_values > eps)
print(f"{n_singular_values=}")

# %%
# Plot the singular values
plt.figure()
plt.plot(singular_values, "o-")
plt.title("Singular Values")
plt.xlabel("Component number")
plt.ylabel("Singular value")
plt.grid(True)
plt.show()


# %%
plt.figure()
plt.semilogy(singular_values, "o-")
plt.title("Singular Values (Log Scale)")
plt.xlabel("Component number")
plt.ylabel("Singular value (log scale)")
plt.grid(True)
plt.show()


# %%
