import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_excel("shopping_trends.xlsx")

freq_map = {
    'Annually': 1,
    'Quarterly': 2,
    'Monthly': 3,
    'Fortnightly': 4,
    'Bi-Weekly': 4,
    'Weekly': 5
}

data['Frequency_num'] = data['Frequency of Purchases'].map(freq_map)
data = data.dropna(subset=['Frequency_num', 'Purchase Amount (USD)'])

X = data[['Purchase Amount (USD)', 'Frequency_num']]

kmeans = KMeans(n_clusters=3, random_state=42)
data['Value_Segment'] = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_[:, 0]
labels = ['Low', 'Medium', 'High']
cluster_map = {i: labels[idx] for i, idx in enumerate(centers.argsort())}
data['Value_Segment'] = data['Value_Segment'].map(cluster_map)

data['Satisfaction'] = data['Review Rating'].apply(lambda x: 'Satisfied' if x > 3.5 else 'Unsatisfied')

print(data[['Customer ID', 'Purchase Amount (USD)', 'Frequency of Purchases', 'Value_Segment', 'Satisfaction']])

data.to_csv("Customer_Segments.csv", index=False)



