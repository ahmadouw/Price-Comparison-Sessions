import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import scipy.stats as stats

exclude_columns = ['visitorId', 'referrerType', 'visitorType', 'category'] 

def preprocess_data(df):
    #select only columns to standardize and apply to the df
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    filtered_df = df[columns_to_scale]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_df)
    return scaled_data, filtered_df

def preprocess_data_z_score(df, threshold):
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    columns_for_zscore = ['npageViews', 'nproducts', 'timeSpent', 'nsearches', 'avgLeadSum']

    scaler = StandardScaler()
    z_scores_selected = scaler.fit_transform(df[columns_for_zscore])

    #create a mask for outliers based on the z-score threshold
    outlier_mask = (np.abs(z_scores_selected) > threshold).any(axis=1)

    #filter the DataFrame based on the z-score threshold
    df= df[~outlier_mask]

    filtered_df = df[columns_to_scale]
    #scale all columns to mean 0 and sd 1 (standardize)
    scaler = StandardScaler()
    standard_stats = scaler.fit_transform(filtered_df)
    return standard_stats, filtered_df

def plot_corr_heatmap(df):
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    filtered_df = df[columns_to_scale]
    correlation_matrix = filtered_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Visit Statistics Variables')
    plt.show()

def find_opt_k(data, params):
    n_clusters_range = params['n_clusters_range']
    batch_size = params['batch_size']
    max_iter = params['max_iter']
    random_state = params['random_state']
    init = params['init']
    
    sse, silhouettes, ch_scores = [], [], []
    for i in n_clusters_range:
        kmeans = MiniBatchKMeans(n_clusters=i, batch_size=batch_size, max_iter=max_iter, init=init, n_init=1, random_state=random_state)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        if i > 1:
            silhouettes.append(silhouette_score(data, kmeans.labels_, sample_size=params['silhouette_sample_size']))
            ch_scores.append(calinski_harabasz_score(data, kmeans.labels_))
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    #plot SSE
    ax[0].plot(n_clusters_range, sse, marker='o')
    ax[0].set_title('Elbow Method for Optimal k')
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('SSE')

    #plot silhouette scores
    ax[1].plot(n_clusters_range[1:], silhouettes, marker='o')
    ax[1].set_title('Silhouette Score for Optimal k')
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Silhouette Score')

    #plot Calinski-Harabasz scores
    ax[2].plot(n_clusters_range[1:], ch_scores, marker='o')
    ax[2].set_title('Calinski-Harabasz Score for Optimal k')
    ax[2].set_xlabel('Number of clusters')
    ax[2].set_ylabel('Calinski-Harabasz Score')

    plt.tight_layout()
    plt.show()
    return silhouettes

def perform_clustering(standard_stats, filtered_df, params,k):
    #extract parameters
    batch_size = params['batch_size']
    max_iter = params['max_iter']
    random_state = params['random_state']
    
    #perform clustering with the maximum silhouette score
    kmeans_final = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, max_iter=max_iter,
                                   init='k-means++', n_init=1, random_state=random_state)
    kmeans_final.fit(standard_stats)
    
    #retrieve cluster labels and cluster centers
    cluster_labels = kmeans_final.labels_
    cluster_centers = kmeans_final.cluster_centers_
    
    #sum of squared distances of samples to their closest cluster center
    inertia = kmeans_final.inertia_
    filtered_df['cluster'] = cluster_labels

    return cluster_labels, cluster_centers, inertia, filtered_df
    
def cluster_means_heatmap(data):
    #calculate cluster means
    cluster_means = data.groupby('cluster').mean()
    scaler = StandardScaler()
    cluster_means_standardized = scaler.fit_transform(cluster_means)
    cluster_means_standardized = pd.DataFrame(cluster_means_standardized, columns=cluster_means.columns)

    #standardize overall mean for consistency with cluster means
    overall_mean = data.drop('cluster', axis=1).mean()
    overall_mean_standardized = scaler.transform([overall_mean])
    overall_mean_standardized = pd.DataFrame(overall_mean_standardized, columns=cluster_means.columns)

    num_samples_per_cluster = data['cluster'].value_counts().sort_index()
    total_samples = len(data)

    #concatenate cluster_means_standardized with overall_mean_standardized
    combined_means = pd.concat([cluster_means_standardized, overall_mean_standardized])

    #plot cluster heatmap with standardized data and absolute values for annotation
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(combined_means.T, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)


    xticks_labels = [f"Cluster {i} ({num_samples_per_cluster[i]/total_samples:.2f})" for i in range(len(num_samples_per_cluster))] + ['Overall ({:.2f})'.format(total_samples/total_samples)]
    plt.xticks(ticks=range(len(xticks_labels)), labels=xticks_labels, rotation=45)
    plt.title('Cluster Heatmap of Feature Means (Standardized)')
    plt.xlabel('Cluster / Overall Mean')
    plt.ylabel('Feature')
    plt.show()

def plot_standardized_proportions(merged_df, property):
    #calculate the proportions of each property per cluster and standardize them to highlight differences in distributions across clusters
    #calculate counts per cluster and property
    counts_by_cluster = merged_df.groupby('cluster')[property].value_counts().unstack(fill_value=0)

    #calculate total observations per cluster
    total_counts_by_cluster = merged_df['cluster'].value_counts()

    #calculate proportions per cluster and property
    proportions_by_cluster = counts_by_cluster.div(total_counts_by_cluster, axis=0)

    #calculate overall proportions
    overall_proportions = proportions_by_cluster.mean(axis=0)
    overall_proportions.name = 'Overall'


    #standardize the proportions
    standardized_proportions = (proportions_by_cluster - proportions_by_cluster.mean()) / proportions_by_cluster.std()

    #create a heatmap for standardized proportions
    plt.figure(figsize=(12, 8))
    sns.heatmap(standardized_proportions.T, cmap='coolwarm', annot=True, fmt=".2f")
    plt.xlabel('Cluster')
    plt.ylabel(property)
    plt.title(f'Standardized Proportions of {property.capitalize()}s in Clusters (Including Overall)')
    plt.show()


def evaluate_classifiers(X, y, params):
    #define ML classifiers
    classifiers = [
        ('XGB', XGBClassifier(n_jobs=1)), 
        ('Random Forest', RandomForestClassifier(n_jobs=1)), 
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('SVM', SVC()),
        ('Logistic Regression', LogisticRegression())
    ]

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    #perform 3-fold cross-validation for each classifier
    for name, classifier in classifiers:
        pipeline = Pipeline([
            ('classifier', classifier)
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=params['random_state'])
        cv_accuracy = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(pipeline, X, y, cv=cv, scoring='precision_weighted')
        cv_recall = cross_val_score(pipeline, X, y, cv=cv, scoring='recall_weighted')
        cv_f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')

        #store mean scores for each metric
        accuracy_scores.append((name, np.mean(cv_accuracy)))
        precision_scores.append((name, np.mean(cv_precision)))
        recall_scores.append((name, np.mean(cv_recall)))
        f1_scores.append((name, np.mean(cv_f1)))

    #print the evaluation metrics
    print("Evaluation Metrics:")
    print("====================")
    for metric_name, metric_scores in [("Accuracy", accuracy_scores), ("Precision", precision_scores), ("Recall", recall_scores), ("F1-score", f1_scores)]:
        print(metric_name)
        for name, score in metric_scores:
            print(f"{name}: {score:.4f}")
        print()

def perform_ANOVA(df):
    exclude_columns = ['visitorId', 'referrerType', 'visitorType', 'category','cluster'] 
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    scaled = df[columns_to_scale]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(scaled)
    scaled = pd.DataFrame(scaled, columns=columns_to_scale)
    scaled['cluster'] = df['cluster'].values
    anova_results = {}
    for col in columns_to_scale:
        anova_results[col] = stats.f_oneway(*[scaled[scaled['cluster'] == cluster][col] for cluster in scaled['cluster'].unique()])
    print(f"{'Feature':<22} | {'F-Statistic':<20} | {'p-value':<8}")
    print('-' * 50)

    for feature, results in anova_results.items():
        f_statistic, p_value = results 
        print(f"{feature:<22} | {f_statistic:>20,.2f} | {p_value:.3f}")

#calculate the normalized value counts for a given column in all clusters
def calculate_normalized_counts(column_name, clusters):
    normalized_counts = {}
    for i, cluster in enumerate(clusters):
        normalized_counts[f'Cluster {i}'] = cluster[column_name].value_counts(normalize=True)
    return pd.DataFrame(normalized_counts).fillna(0)

#create principal components of different stats
def apply_PCA(df, components):
    pca = PCA(n_components = components)
    reduced_stats = pca.fit_transform(df)
    
    return reduced_stats