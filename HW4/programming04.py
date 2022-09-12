#---------------------------------------- Parse of input ---------------------------------------#
import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

raw_data = loadarff('breast.w.arff')
# converting data to a pandas DataFrame
df_data = pd.DataFrame(raw_data[0]).dropna()
df_data['Class'].replace({b'malignant': 1, b'benign': 0}, inplace=True)
X, y = df_data.drop(columns='Class'), df_data['Class']

#------------------------------- k= 2,3 clustering model analysis ------------------------------#


def ECR(k):
    ecr = 0
    for c in range(k):
        # points from that cluster
        points = df_data.loc[df_data['Cluster'] == c]
        # most frequent class assigned to that cluster
        max_frequency_class = max(points['Class'].value_counts())
        ecr += (len(points) - max_frequency_class)
    return ecr / k


for k in [2, 3]:
    kmeans = KMeans(n_clusters=k)
    # new column with the cluster to each point
    df_data['Cluster'] = kmeans.fit_predict(X)

    print("For k =", k, "ECR:", ECR(k), "Silhouette:",
          silhouette_score(X, kmeans.labels_, metric='euclidean'))

#--------------------------- Plot of 3-means clustering solution in 2D -------------------------#
kbest = SelectKBest(mutual_info_classif, k=2)
# data with only the selected features
x_transformed = kbest.fit_transform(X, y)
cols = kbest.get_support(indices=True)  # indexes of the selected features
# names of the features
feat1, feat2 = df_data.columns[cols[0]], df_data.columns[cols[1]]

# centroid coordinates with only the selected features
cen_x, cen_y = kmeans.cluster_centers_[
    :, cols[0]], kmeans.cluster_centers_[:, cols[1]]

# getting colors for each point according to cluster color
colors = ['#DF2020', '#81DF20', '#2095DF']
mapped_colors = np.array([colors[i] for i in df_data['Cluster']])

plt.figure(figsize=(10, 4))

# plot of the points belonging to class 0
plt.scatter(x_transformed[y == 0][:, 0], x_transformed[y == 0]
            [:, 1], c=mapped_colors[y == 0], s=40, marker='o')
# plot of the points belonging to class 1
plt.scatter(x_transformed[y == 1][:, 0], x_transformed[y == 1]
            [:, 1], c=mapped_colors[y == 1], s=20, marker='^')
plt.scatter(cen_x, cen_y, c=colors, s=100, alpha=0.5)  # plot of the centroids

plt.xlabel(feat1)
plt.ylabel(feat2)
plt.show()
