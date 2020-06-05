import pandas as pd

df = pd.read_csv('SisterCitiesData.csv', delimiter='\t')

total_capitals = df.iscapital.sum()
possible_configs = [c for c in df.columns if c.startswith('distance')]
def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec)

# Study the case with encoding distance = 4 and 6 clusters
for config in possible_configs:
    in_group_df = df.groupby(['iscapital', config]).count()['Id'].to_frame().reset_index()
    in_group_capitals_df = in_group_df.loc[in_group_df.iscapital == 1]
    cluster_id = in_group_capitals_df.loc[in_group_capitals_df['Id'].argmax()][config]

    # Get the values within the cluster
    in_cluster = df.loc[df[config] == cluster_id]
    capital_set = in_cluster.groupby(in_cluster.iscapital).count()['Id'].to_frame().reset_index()
    total_values = capital_set['Id'].sum()
    cluster_capitals = capital_set.loc[capital_set.iscapital == 1]['Id'][1]

    # Compute degree counts
    degree_top_n = df.sort_values('Degree', ascending=False)[:total_values].iscapital
    degree_capitals = degree_top_n.sum()
    pagerank_top_n = df.sort_values('pageranks', ascending=False)[:total_values].iscapital.sum()
    pagerank_capitals = pagerank_top_n.sum()

    # Prepare the metrics
    # Recall
    cluster_rec = cluster_capitals / total_values
    degree_rec = degree_capitals / total_values
    pagerank_rec = pagerank_capitals / total_values

    # Precision
    cluster_prec = cluster_capitals / total_capitals
    degree_prec = degree_capitals / total_capitals
    pagerank_prec = pagerank_capitals / total_capitals

    # F1-Score
    cluster_f1 = f1_score(cluster_prec, cluster_rec)
    degree_f1 = f1_score(degree_prec, degree_rec)
    pagerank_f1 = f1_score(pagerank_prec, pagerank_rec)

    print('The {} proportion of capitals is {} for the top cluster, {} for top-n degree, {} for top-n pagerank with n = {}'.format(
        config, 
        cluster_f1,
        degree_f1,
        pagerank_f1,
        total_values
        ))

