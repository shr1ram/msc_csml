# %%
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# %%
def load_data(filename):
    data = np.loadtxt(filename)
    y_raw = data[:, 0]
    X = data[:, 1:]

    y = np.ones_like(y_raw)
    y[y_raw == 1] = -1
    # map 3->1, 1->-1
    return X, y

def build_W(X):
    # calculate distances, find indices for 3NN for each point
    dists = cdist(X, X, 'euclidean')
    sorted_indices = np.argsort(dists, axis=1)
    nn_indices = sorted_indices[:, 1:4] 

    # form directed adjacency matrix A
    m = X.shape[0]
    A = np.zeros((m, m))
    for i in range(m):
        A[i, nn_indices[i]] = 1

    # convert to undirected (as required) adjacency matrix W
    W = np.maximum(A, A.T)
    return W

def build_D(W):
    row_sums = np.sum(W, axis=1)
    D = np.diag(row_sums)
    return D



# %%
# Testing

X, y = load_data('dtrain13_50.dat')
W = build_W(X)
D = build_D(W)
L = D - W
L_pinv = np.linalg.pinv(L)

# %%
def LaplacianInterpolation(L, y, labeled_idx, unlabeled_idx):
    preds = interpolate_laplacian(L, y, labeled_idx, unlabeled_idx)
    mistakes = preds[unlabeled_idx] != y[unlabeled_idx]
    err = np.mean(mistakes)
    return err

def interpolate_laplacian(L, y, labeled_idx, unlabeled_idx):

    # from paper, we are solving f_u = (D_uu - W_uu)^-1*W_ul*f_l
    # which translates to u_u = L_uu^-1*W_ul*u_l
    # D is a diagonal matrix, so D_ul = 0, so 0 - W_ul = L_ul
    # u_u*Luu = L_ul*u_l

    m = L.shape[0]
    u_l = y[labeled_idx]
    L_uu = L[np.ix_(unlabeled_idx, unlabeled_idx)] # unlabelled - unlabelled interaction weights
    L_ul = L[np.ix_(unlabeled_idx, labeled_idx)] # unlabelled - labelled interaction weights

    rhs = -L_ul @ u_l
    u_u = np.linalg.solve(L_uu, rhs)

    # build continuous prediction vector and fill
    v = np.zeros(m)
    v[labeled_idx] = u_l
    v[unlabeled_idx] = u_u

    return np.sign(v)

def LaplacianKernelInterpolation(L_pinv, y, labeled_idx, unlabeled_idx):
    preds = interpolate_laplacian_kernel(L_pinv, y, labeled_idx)
    mistakes = preds[unlabeled_idx] != y[unlabeled_idx]
    err = np.mean(mistakes)
    return err

def interpolate_laplacian_kernel(L_pinv, y, labeled_idx):

    # build kernel marix from labeled indices of Laplacian pseudoinverse
    kernel = L_pinv[np.ix_(labeled_idx, labeled_idx)]

    # compute weights alpha
    alpha = np.linalg.pinv(kernel) @ y[labeled_idx]

    # compute continuous predictions
    v = alpha @ L_pinv[labeled_idx, :]

    # discretise prediction vector and return
    return np.sign(v)

# %%
datasets = [
    "dtrain13_50.dat",
    "dtrain13_100.dat", 
    "dtrain13_200.dat", 
    "dtrain13_400.dat"
]
lList = [1, 2, 4, 8, 16]

meanErrLI_table = np.zeros((len(datasets), len(lList)))
meanErrLKI_table = np.zeros((len(datasets), len(lList)))
stdErrLI_table = np.zeros((len(datasets), len(lList)))
stdErrLKI_table = np.zeros((len(datasets), len(lList)))
for dsIdx, ds in enumerate(datasets):

    X, y = load_data(ds)

    W = build_W(X)
    D = build_D(W)
    L = D - W
    L_pinv = np.linalg.pinv(L)

    for lIdx, l in enumerate(lList):
        errLI = []
        errLKI = []
        for _ in range(20):

            # identify indices of each class
            idx_class1 = np.where(y == -1)[0]
            idx_class2 = np.where(y == 1)[0]

            # sample l times without replacement from each class and build labeled indices vector
            samples_class1 = np.random.choice(idx_class1, l, replace=False)
            samples_class2 = np.random.choice(idx_class2, l, replace=False)
            labeled_idx = np.concatenate([samples_class1, samples_class2])

            # build unlabeled indices vector
            m = len(y)
            mask = np.zeros(m, dtype=bool)
            mask[labeled_idx] = True
            unlabeled_idx = np.where(~mask)[0]

            errLI += [LaplacianInterpolation(L, y, labeled_idx, unlabeled_idx)]
            errLKI += [LaplacianKernelInterpolation(L_pinv, y, labeled_idx, unlabeled_idx)]
        meanErrLI_table[dsIdx][lIdx] = np.mean(errLI)
        meanErrLKI_table[dsIdx][lIdx] = np.mean(errLKI)
        stdErrLI_table[dsIdx][lIdx] = np.std(errLI)
        stdErrLKI_table[dsIdx][lIdx] = np.std(errLKI)


# %%
mList = [50, 100, 200, 400]

# Laplacian Interpolation

formatted_data = np.empty((len(mList), len(lList)), dtype=object)
for i in range(len(mList)):
    for j in range(len(lList)):
        mean_val = meanErrLI_table[i, j]
        std_val = stdErrLI_table[i, j]
        formatted_data[i, j] = f"{mean_val:.3f} ± {std_val:.3f}"

df_LI = pd.DataFrame(
    formatted_data, 
    index=mList, 
    columns=lList
)
df_LI.index.name = "v # of data points per label"
df_LI.columns.name = "> # of known labels (per class)"

# Laplacian Kernel Interpolation

formatted_data_lki = np.empty(meanErrLKI_table.shape, dtype=object)

formatted_data = np.empty((len(mList), len(lList)), dtype=object)
for i in range(len(mList)):
    for j in range(len(lList)):
        mean_val = meanErrLKI_table[i, j]
        std_val = stdErrLKI_table[i, j]
        formatted_data[i, j] = f"{mean_val:.3f} ± {std_val:.3f}"

df_LKI = pd.DataFrame(
    formatted_data, 
    index=mList, 
    columns=lList
)
df_LKI.index.name = "v # of data points per label"
df_LKI.columns.name = "> known labels (per class)"

# Print Results

pd.set_option('display.width', 1000)

print("=== Laplacian Interpolation (LI) ===")
print(df_LI)
print("\n")

print("=== Laplacian Kernel Interpolation (LKI) ===")
print(df_LKI)


