import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#analysis of users across clusters
#number of unique users
def overall_unique_users(df):
    return df['visitorId'].nunique()

#number of unique users per cluster
def unique_users_per_cluster(df):
    unique_users_in_clusters = {}
    for i in df['cluster'].unique():
        unique_users_in_clusters[i] = df.loc[df['cluster'] == i, 'visitorId'].nunique()
    return unique_users_in_clusters

#number of users appearing in multiple clusters
def users_in_multiple_clusters(df):
    users_in_multiple_clusters = df.groupby('visitorId')['cluster'].nunique()
    users_in_multiple_clusters = users_in_multiple_clusters[users_in_multiple_clusters > 1]
    return users_in_multiple_clusters

#number of users appearing in exactly one cluster
def users_with_visits_in_one_cluster(df):
    users_single_cluster = df.groupby('visitorId')['cluster'].nunique()
    users_single_cluster = users_single_cluster[users_single_cluster == 1]
    
    return users_single_cluster

#number of users with multiple visits within one cluster only
def users_with_multiple_visits_in_one_cluster(df):
    users_with_multiple_visits = df.groupby('visitorId').size()
    users_with_multiple_visits = users_with_multiple_visits[users_with_multiple_visits > 1]
    
    users_single_cluster = df.groupby('visitorId')['cluster'].nunique()
    users_single_cluster = users_single_cluster[users_single_cluster == 1]
    
    return users_with_multiple_visits.index.intersection(users_single_cluster.index)


#display user stats
def user_stats(df):
    unique_users = overall_unique_users(df)
    print(f"Overall proportion of unique users: {unique_users/len(df):.2f}")

    #compute the number of unique users in each cluster
    unique_users_in_clusters = unique_users_per_cluster(df)
    for cluster, count in unique_users_in_clusters.items():
        print(f"Proportion of unique users in cluster {cluster}: {count/len(df['cluster']==cluster):.2f}")

    #identify the number of users with visits in only one cluster
    one_cluster = users_with_visits_in_one_cluster(df)
    print(f"Proportion of users with visits in only one cluster: {len(one_cluster)/unique_users:.2f}")

    #identify users who have more than one visit and appear in only one cluster
    multiple_visits_in_one_cluster = users_with_multiple_visits_in_one_cluster(df)
    print(f"Proportion of users with multiple visits in only one cluster: {len(multiple_visits_in_one_cluster)/unique_users:.2f}")

    #identify users who appear in multiple clusters
    multiple_clusters = users_in_multiple_clusters(df)
    print(f"Proportion of users appearing in multiple clusters: {len(multiple_clusters)/unique_users:.2f}")


def calculate_proportions(df, fields):
    #calculate proportions with defined threshold
    proportions = {}
    for field in fields:
        if field == 'nproducts':
            proportion = (df[field] > 1).mean()
        else:
            proportion = (df[field] > 0).mean()
        proportions[field] = proportion
    return proportions

def plot_combined_proportions(cluster_dfs, clusters, fields):
    all_proportions = {}
    
    #calculate proportions for each cluster
    for cluster_df, cluster_id in zip(cluster_dfs, clusters):
        proportions = calculate_proportions(cluster_df, fields)
        all_proportions[cluster_id] = proportions
    
    proportions_df = pd.DataFrame(all_proportions)
    proportions_df.index.name = 'Cluster'
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(proportions_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Proportions of Sessions with Values Greater Than Threshold Across Clusters')
    plt.xlabel('Fields')
    plt.ylabel('Clusters')
    plt.show()

def calculate_cluster_verification_proportions(df):
    results = {}

    # Q1a: How many sessions involve products priced above a certain threshold (500 Euro)?
    def q1a(df):
        return df['avgLeadSum'] > 500

    # Q1b: How many sessions involve products priced below a certain threshold (100 Euro)?
    def q1b(df):
        return df['avgLeadSum'] < 100

    # Q2: How many sessions last less than five minutes?
    def q2(df):
        return df['timeSpent'] < 300

    # Q3: How many sessions involve viewing fewer than 3 products but still generate at least one lead?
    def q3(df):
        return (df['nproducts'] < 3) & (df['nleads'] > 0)

    # Q4a: How many sessions involve using filters at least five times and generate a lead?
    def q4a(df):
        return (df['nfilters'] >= 5) & (df['nleads'] > 0)

    # Q4b: How many sessions involve using filters at least two times?
    def q4b(df):
        return df['nfilters'] >= 2

    # Q4c: How many sessions involve using no filters?
    def q4c(df):
        return df['nfilters'] == 0

    # Q5a: How many sessions involve using search at least five times and generate a lead?
    def q5a(df):
        return (df['nsearches'] >= 5) & (df['nleads'] > 0)

    # Q5b: How many sessions involve using search at least two times?
    def q5b(df):
        return df['nsearches'] >= 2

    # Q5c: How many sessions involve using no search?
    def q5c(df):
        return df['nsearches'] == 0

    # Q6: How many sessions view more than five products?
    def q6(df):
        return df['nproducts'] > 5

    # Q7a: How many sessions view more than five products?
    def q7a(df):
        return df['nproducts'] > 5

    # Q7b: How many sessions view more than five products but do not generate a lead?
    def q7b(df):
        return (df['nproducts'] > 5) & (df['nleads'] == 0)

    # Q8: How many sessions last more than ten minutes but involve viewing fewer than five products?
    def q8(df):
        return (df['timeSpent'] > 600) & (df['nproducts'] < 5)

    # Q9a: How many sessions last more than ten minutes and do not lead to leads?
    def q9a(df):
        return (df['timeSpent'] > 600) & (df['nleads'] == 0)

    # Q9b: How many sessions last more than twenty minutes and do not lead to leads?
    def q9b(df):
        return (df['timeSpent'] > 1200) & (df['nleads'] == 0)

    # Q10: How many sessions browse more than three third-level categories?
    def q10(df):
        return df['3rdLevelVisits'] > 3

    # Q11: How many sessions browse more than two second-level categories?
    def q11(df):
        return df['2ndLevelVisits'] > 2

    # Q12: How many sessions generate more than one lead?
    def q12(df):
        return df['nleads'] > 1

    questions = {
        'Q1a': q1a,
        'Q1b': q1b,
        'Q2': q2,
        'Q3': q3,
        'Q4a': q4a,
        'Q4b': q4b,
        'Q4c': q4c,
        'Q5a': q5a,
        'Q5b': q5b,
        'Q5c': q5c,
        'Q6': q6,
        'Q7a': q7a,
        'Q7b': q7b,
        'Q8': q8,
        'Q9a': q9a,
        'Q9b': q9b,
        'Q10': q10,
        'Q11': q11,
        'Q12': q12
    }

    clusters = df['cluster'].unique()
    clusters = sorted(clusters)

    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        cluster_results = {}
        for q, func in questions.items():
            cluster_results[q] = func(cluster_data).mean() 
        results[cluster] = cluster_results

    return pd.DataFrame(results).round(2)


def calculate_reported_questions(df, threshold=500):
    results = {}

    # Q1: How many sessions involve leads priced above 500 euros?
    def q1(df, threshold=500):
        return df['avgLeadSum'] > threshold

    # Q2: How many sessions last less than five minutes?
    def q2(df):
        return df['timeSpent'] < 300

    # Q3: How many sessions involve viewing fewer than 3 products but still generate at least one lead?
    def q3(df):
        return (df['nproducts'] < 3) & (df['nleads'] > 0)

    # Q4: How many sessions involve using filters at least three times and generate a lead?
    def q4(df):
        return (df['nfilters'] >= 3) & (df['nleads'] > 0)

    # Q5: How many sessions involve using search at least three times and generate a lead?
    def q5(df):
        return (df['nsearches'] >= 3) & (df['nleads'] > 0)

    # Q6: How many sessions view more than five products?
    def q6(df):
        return df['nproducts'] > 5

    # Q7: How many sessions last more than ten minutes but involve viewing fewer than five products?
    def q7(df):
        return (df['timeSpent'] > 600) & (df['nproducts'] < 5)

    # Q8: How many sessions last more than ten minutes and do not generate a lead?
    def q8(df):
        return (df['timeSpent'] > 600) & (df['nleads'] == 0)

    # Q9: How many sessions browse more than three third-level categories?
    def q9(df):
        return df['3rdLevelVisits'] > 3

    # Mapping questions to their corresponding functions
    questions = {
        'Q1': q1,
        'Q2': q2,
        'Q3': q3,
        'Q4': q4,
        'Q5': q5,
        'Q6': q6,
        'Q7': q7,
        'Q8': q8,
        'Q9': q9
    }

    clusters = df['cluster'].unique()
    clusters = sorted(clusters)

    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        cluster_results = {}
        for q, func in questions.items():
            cluster_results[q] = func(cluster_data).mean()  # Calculate the proportion
        results[cluster] = cluster_results

    result_df = pd.DataFrame(results)
    return result_df.round(2)  # Round to 2 decimal places

def shared_users(df):
    #create a contingency table of visitorId and clusters
    crosstab_df = pd.crosstab(df['visitorId'], df['cluster'])

    #shared users count
    shared_users_df = pd.DataFrame(0, index=crosstab_df.columns, columns=crosstab_df.columns)

    #calculate the number of shared users between each pair of clusters
    for cluster1 in crosstab_df.columns:
        for cluster2 in crosstab_df.columns:
            if cluster1 != cluster2:
                shared_users = crosstab_df[(crosstab_df[cluster1] > 0) & (crosstab_df[cluster2] > 0)].shape[0]
                shared_users_df.loc[cluster1, cluster2] = shared_users

    #calculate the unique visitor counts for each cluster
    unique_visitor_counts = crosstab_df.sum(axis=0)

    #normalize the shared users count by the number of unique users in each cluster
    normalized_shared_users_df = shared_users_df.div(unique_visitor_counts, axis=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_shared_users_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Normalized Shared Users Between Clusters')
    plt.xlabel('Target Cluster')
    plt.ylabel('Source Cluster')
    plt.show()

def mean_price(df):
    #calculate mean price of viewed products for all sessions
    action_df = df[df['type'] == 'action']
    filtered_df = action_df.dropna(subset=['productViewPrice'])
    average_price_per_visit = filtered_df.groupby('idVisit')['productViewPrice'].mean().reset_index()
    merged_visits = average_price_per_visit.merge(action_df[['idVisit', 'cluster']].drop_duplicates(), on='idVisit')

    def remove_outliers(df, column):
        #use the inter-quartile range to remove outliers
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    cleaned_data = merged_visits.groupby('cluster').apply(remove_outliers, 'productViewPrice').reset_index(drop=True)
    
    #standardize price values
    scaler = StandardScaler()
    cleaned_data['standardizedProductViewPrice'] = scaler.fit_transform(cleaned_data[['productViewPrice']])


    plt.figure(figsize=(12, 8))
    sns.boxplot(x='cluster', y='standardizedProductViewPrice', data=cleaned_data)
    plt.title('Standardized average Product View Prices per Visit by Cluster (Outliers Removed)')
    plt.xlabel('Cluster')
    plt.ylabel('Standardized average Product View Price')
    plt.show()
