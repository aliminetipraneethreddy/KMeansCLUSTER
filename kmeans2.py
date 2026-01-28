import pandas as pd

path='Wholesale customers data.csv'
df=pd.read_csv(path)
df.head()


"""# Task
Perform customer segmentation on the provided `df` DataFrame by cleaning the data, preprocessing it by scaling the features, determining the optimal number of clusters (K) using methods like the Elbow Method or Silhouette Score, building a K-Means clustering model to assign customer segments, visualizing these segments, and profiling each cluster to provide business interpretations and strategies. Additionally, assess the stability of the clustering solution by rerunning the process with a different `random_state` value. The final output should summarize the findings, including segment characteristics, proposed business strategies, and a discussion on stability. The analysis should focus on `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, and `Delicassen` columns.

## Data Exploration and Cleaning

### Subtask:
Inspect the dataset, identify and select features representing customer purchasing behavior, and remove or ignore irrelevant columns. This includes `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, and `Delicassen` columns.

**Reasoning**:
To begin the data exploration and cleaning subtask, I will create a new DataFrame 'X' containing only the specified columns related to customer purchasing behavior and then display its first few rows to confirm the selection.
"""

X = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
X.head()

"""## Data Preprocessing

### Subtask:
Prepare the selected numerical features for clustering by checking for missing values and scaling the features.

**Reasoning**:
The first instruction is to check for missing values in the `X` DataFrame. I will use `.isnull().sum()` to identify them.
"""

print('Missing values in X:')
print(X.isnull().sum())

"""**Reasoning**:
Since no missing values were found, the next step is to scale the features using `StandardScaler` to prepare the data for clustering. I will import it, instantiate it, and then fit and transform the data, storing it in `X_scaled`.


"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X_scaled created and transformed using StandardScaler.")
print("First 5 rows of scaled data:\n", X_scaled[:5])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X_scaled)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X_scaled)

y_kmeans

plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], s=100, c='pink', label='Cluster 4')

# Optional: Plotting cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='yellow', marker='*', label='Centroids')

plt.title("Clusters of Customers")
plt.xlabel("Fresh")
plt.ylabel("Milk")
plt.legend()
plt.show()

