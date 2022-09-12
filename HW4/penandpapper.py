import numpy as np
from scipy.stats import multivariate_normal

cov1 = [[1, 0], [0, 1]]
cov2 = [[2, 0], [0, 2]]
e1 = [[2], [4]]
e2 = [[-1], [-4]]

e11 = [2, 4]
e21 = [-1, -4]

p1 = 0.7
p2 = 0.3

x1 = [[2], [4]]
x2 = [[-1], [-4]]
x3 = [[-1], [2]]
x4 = [[4], [0]]

x11 = [2, 4]
x21 = [-1, -4]
x31 = [-1, 2]
x41 = [4, 0]


c1 = []
c2 = []

x = [x11, x21, x31, x41]
posteriors1 = []
posteriors2 = []

def joint_prob(x, c):
    if c == 1:
        likelihood = multivariate_normal(e11, cov1).pdf(x)
        print("likelihood for", x, likelihood)
        joint = p1 * likelihood
        print("p(x| c1)*p(c1)", joint)
    else:
        likelihood = multivariate_normal(e21, cov2).pdf(x)
        print("likelihood for", x, likelihood)
        joint = p2 * likelihood
        print("p(x| c2)*p(c2)", joint)
    return joint


def normalized_posterior(x):
    jp1 = joint_prob(x, 1)
    jp2 = joint_prob(x, 2)

    pc1_x = jp1 / (jp1 + jp2)
    pc2_x = jp2 / (jp1 + jp2)
    posteriors1.append(pc1_x)
    posteriors2.append(pc2_x)
    print("normalized posterior p(c1 | x) = ", pc1_x)
    print("normalized posterior p(c2 | x) = ", pc2_x)
    if pc1_x > pc2_x:
        c1.append(x)
        print("x pertence ao cluster 1")
    else:
        c2.append(x)
        print("x pertence ao cluster 2")

# Expectation step


def expectation():
    print("X1")
    normalized_posterior(x11)
    print("X2")
    normalized_posterior(x21)
    print("X3")
    normalized_posterior(x31)
    print("X4")
    normalized_posterior(x41)

# Maximization Step


def cov_cell_update(i, j, c):
    soma = 0
    den = 0
    if (c == 1):
        for n in range(len(x)):
            soma += posteriors1[n] * \
                (x[n][i][0] - e1[i][0]) * (x[n][j][0] - e1[j][0])
            den += posteriors1[n]
    else:
        for n in range(len(x)):
            soma += posteriors2[n] * \
                (x[n][i][0] - e2[i][0]) * (x[n][j][0] - e2[j][0])
            den += posteriors2[n]
    return soma / den


def maximization_e():
    miu1 = [[0], [0]]
    den1 = 0
    for i in range(0, len(x)):
        miu1 += np.multiply(posteriors1[i], x[i])
        den1 += posteriors1[i]
    miu1 = np.multiply(1/den1, miu1)
    miu2 = [[0], [0]]
    den2 = 0
    for i in range(0, len(x)):
        miu2 += np.multiply(posteriors2[i], x[i])
        den2 += posteriors2[i]
    miu2 = np.multiply(1/den2, miu2)

    return miu1, miu2


def maximization_c():
    cov_1 = [[0, 0], [0, 0]]
    cov_2 = [[0, 0], [0, 0]]

    for row in range(2):
        for column in range(2):
            cov_1[row][column] = cov_cell_update(row, column, 1)

    for row in range(2):
        for column in range(2):
            cov_2[row][column] = cov_cell_update(row, column, 2)

    return cov_1, cov_2


def priors_update():
    soma1, soma2, den = 0, 0, 0
    for i in range(len(x)):
        soma1 += posteriors1[i]
        soma2 += posteriors2[i]
    den = soma1 + soma2
    return (soma1/den), (soma2/den)


expectation()

x = [x1, x2, x3, x4]

e1, e2 = maximization_e()
cov1, cov2 = maximization_c()

p1, p2 = priors_update()


print("New e1 ", e1, " New e2 ", e2)
print("New cov1 ", cov1, "New cov2", cov2)
print("New p1 ", p1, " New p2 ", p2)


def silhouete_x(x):
    a = 0
    b = 0
    if (x in c1):
        for x_1 in c1:
            a += np.linalg.norm(np.subtract(x, x_1))
        for x_2 in c2:
            b += np.linalg.norm(np.subtract(x, x_2))
        a = a / (len(c1) - 1)
        b = b / len(c2)
    else:
        for x_2 in c2:
            a += np.linalg.norm(np.subtract(x, x_2))
        for x_1 in c1:
            b += np.linalg.norm(np.subtract(x, x_1))
        if a != 0:
            a = a / (len(c2) - 1)
        if b != 0:
            b = b / len(c1)

    if (b > a):
        return 1 - a/b
    else:
        return b/a - 1


def silhouette_ck(k):
    soma = 0
    if k == 1:
        for x in c1:
            soma += silhouete_x(x)
            print("Silhouette x", silhouete_x(x))
        soma = soma / len(c1)
    else:
        for x in c2:
            soma += silhouete_x(x)
            print("Silhouette x", silhouete_x(x))
        soma = soma / len(c2)
    return soma


s1 = silhouette_ck(1)
print("Silhouette c1", s1)
s2 = silhouette_ck(2)
print("Silhouette c2", s2)

print("Silhouette solution")
print((s1 + s2) / 2)
