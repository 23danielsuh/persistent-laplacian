import dionysus as d
import numpy as np
import scipy.io


def schur_comp(A, ind):
    (l, w) = A.shape
    nind = np.setdiff1d(np.arange(w), ind)
    Saa = A[np.ix_(ind, ind)]
    Sab = A[np.ix_(ind, nind)]
    Sba = A[np.ix_(nind, ind)]
    Sbb = A[np.ix_(nind, nind)]

    # Compute S = Saa - Sab * pinv(Sbb) * Sba
    Sbb_pinv = np.linalg.lstsq(Sbb, Sba, rcond=None)[0]
    S = Saa - np.matmul(Sab, Sbb_pinv)
    return S


def persistLap(B1, B2, Gind):
    upLaplacian = B1.dot(B1.T)
    if np.all(B2 == 0):
        upL = schur_comp(upLaplacian, Gind)
        pL = upL
    else:
        downLaplacian = B2.T.dot(B2)
        upL = schur_comp(upLaplacian, Gind)
        pL = upL + downLaplacian
    return (pL, upL)


# The followings are input data
filename = "Point.txt"  # input a point cloud dataset;
data = np.genfromtxt(filename, delimiter=" ")
tK = 1
# threshold for simplicial complex K
tL = 1.5
# threshold for simplicial complex L
end = 1.5
# largest threshold for building the Vietoris-Rips complex
q = 2
# dimension for the persistent Laplacian

# Building a Vietoris-Rips complex
f = d.fill_rips(data, q + 1, end)

count = 0
for s in f:
    count = count + 1
shape = (count, count)
Bw = np.zeros(shape, dtype=int)


if q == 0:  # If q=0, returns only the 1-boundary matrix of L
    q1_column_index = []
    q1_row_index = []
    Kind = 0

    for s in f:
        ind = f.index(s)
        dim = s.dimension()
        if (dim == q + 1) and (s.data <= tL):
            q1_column_index.append(ind)
            sign = 1
            for sb in s.boundary():
                rind = f.index(sb)
                Bw[rind, ind] = sign
                sign = -1 * sign
        if (dim == q) and (s.data <= tL):
            q1_row_index.append(ind)
            if s.data <= tK:
                Kind = Kind + 1

    Bq1 = Bw[:, q1_column_index]
    Bq1 = Bq1[q1_row_index, :]
    mat = np.matrix(Bq1)
    scipy.io.savemat("q1Boundary.mat", mdict={"Bq1": mat}, do_compression=True)
else:  # If q>0, returns both the q-boundary matrix of K and the (q+1)-boundary matrix of L
    q_column_index = []
    q_row_index = []
    q1_column_index = []
    q1_row_index = []
    Kind = 0

    for s in f:
        ind = f.index(s)
        dim = s.dimension()
        if (dim == q + 1) and (s.data <= tL):
            q1_column_index.append(ind)
            sign = 1
            for sb in s.boundary():
                rind = f.index(sb)
                Bw[rind, ind] = sign
                sign = -1 * sign
        if (dim == q) and (s.data <= tL):
            q1_row_index.append(ind)
            if s.data <= tK:
                q_column_index.append(ind)
                sign = 1
                for sb in s.boundary():
                    rind = f.index(sb)
                    Bw[rind, ind] = sign
                    sign = -1 * sign
                Kind = Kind + 1
        if (dim == q - 1) and (s.data <= tK):
            q_row_index.append(ind)

    Bq1 = Bw[:, q1_column_index]
    Bq1 = Bq1[q1_row_index, :]
    Bq = Bw[:, q_column_index]
    Bq = Bq[q_row_index, :]

    B1 = Bq.astype(np.float32)
    B2 = Bq1.astype(np.float32)

    (_, num_K) = B1.shape

    k = 10  # number of eigenvalues we want to calculate
    (pL, _) = persistLap(B2, B1, np.arange(num_K))
    eigenvalues = np.linalg.eigvalsh(pL)
    sorted_eigenvalues = np.sort(eigenvalues)
    top_k_eigenvalues = sorted_eigenvalues[:k]

    print("eigenvalues:")
    print(top_k_eigenvalues)
