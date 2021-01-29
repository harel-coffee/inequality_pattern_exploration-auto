import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
#from pyclustering.cluster.kmedoids import kmedoids

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def evaluate_silhouette(X, range_n_clusters, method='kmeans', plot=True, affinity=None, 
                        X_distance=None, metric=None):
    silhouettes = []

    for n_clusters in range_n_clusters:
        print(n_clusters, sep=' ', end='', flush=True)
        #print(method, n_clusters)
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if method=='kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=10, max_iter=150)
            cluster_labels = clusterer.fit_predict(X)
        if method=='kmdeoids':
            if metric:
                clusterer = KMedoids(n_clusters=n_clusters, random_state=10, init='k-medoids++', metric=metric, max_iter=150)
            else:
                clusterer = KMedoids(n_clusters=n_clusters, random_state=10, init='k-medoids++', max_iter=150)
            cluster_labels = clusterer.fit_predict(X)
        if method=='agg_average':
            if affinity:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average').fit(X_distance)
                cluster_labels = clusterer.labels_
            else:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
                cluster_labels = clusterer.fit_predict(X)
        if method=='agg_complete':
            if affinity:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete').fit(X_distance)
                cluster_labels = clusterer.labels_
            else:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
                cluster_labels = clusterer.fit_predict(X)
        if method=='GMM':
            clusterer = GMM(n_components=n_clusters).fit(X)
            cluster_labels = clusterer.predict(X)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
#         print("For n_clusters =", n_clusters,
#               "The average silhouette_score is :", silhouette_avg)
        silhouettes.append(silhouette_avg)

        if plot:
            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(6,3)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
        
    return dict(zip(range_n_clusters, silhouettes))
    
def plot_silhouette(X, range_n_clusters):

    print('start kmeans')
    res_kmeans = evaluate_silhouette(X, range_n_clusters, method='kmeans', plot=False)
    print('start kmedoids')
    res_kmedoids = evaluate_silhouette(X, range_n_clusters, method='kmdeoids', plot=False)
    print('start agglomerative complete')
    res_agg_complete = evaluate_silhouette(X, range_n_clusters, method='agg_complete', plot=False)
    print('start agglomerative average')
    res_agg_average = evaluate_silhouette(X, range_n_clusters, method='agg_average', plot=False)
    print('start GMM')
    res_GMM = evaluate_silhouette(X, range_n_clusters, method='GMM', plot=False)
    print('finished evaluating sillhouette')
    compare_cluster = pd.DataFrame()

    methods = ['kmeans', 'kmedoids', 'agglomerative-complete', 'agglomerative-average', 'GMM']

    for i, res in enumerate([res_kmeans, res_kmedoids, res_agg_complete, res_agg_average, res_GMM]):
        silhouette_df = pd.DataFrame()
        silhouette_df['no_cluster'] = res.keys()
        silhouette_df['avg_silhouette'] = res.values()
        silhouette_df['method'] = methods[i]

        compare_cluster = compare_cluster.append(silhouette_df)

    fig, ax = plt.subplots(figsize=(6,3.5))
    sns.pointplot(x='no_cluster', y='avg_silhouette', data=compare_cluster, hue='method', ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
def plot_silhouette_distance(X, range_n_clusters, metric='cosine'):

    X_distance = pairwise_distances(X, metric=metric)
    
    res_kmedoids = evaluate_silhouette(X, range_n_clusters, method='kmdeoids', metric=metric, plot=False)
    res_agg_complete = evaluate_silhouette(X, range_n_clusters, method='agg_complete',
                                           affinity=True, X_distance=X_distance, plot=False)
    res_agg_average = evaluate_silhouette(X, range_n_clusters, method='agg_average',
                                           affinity=True, X_distance=X_distance, plot=False)

    compare_cluster = pd.DataFrame()

    methods = ['kmedoids', 'agglomerative-complete', 'agglomerative-average']

    for i, res in enumerate([res_kmedoids, res_agg_complete, res_agg_average]):
        silhouette_df = pd.DataFrame()
        silhouette_df['no_cluster'] = res.keys()
        silhouette_df['avg_silhouette'] = res.values()
        silhouette_df['method'] = methods[i]

        compare_cluster = compare_cluster.append(silhouette_df)

    fig, ax = plt.subplots(figsize=(6,3.5))
    sns.pointplot(x='no_cluster', y='avg_silhouette', data=compare_cluster, hue='method', ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
def append_to_gdf(label_used, results_df):
    dtag_gdf = gpd.read_file('DTAG_districts.shp')
    dtag_gdf = dtag_gdf.loc[dtag_gdf['DN']>0]
    dtag_gdf.reset_index(inplace=True)

    results_df['cluster'] = label_used

    for cluster in label_used:
        cluster_df = results_df.loc[results_df['cluster']==cluster]
        cluster_df = cluster_df[cluster_df.columns[:23]]
        
        avrg_profit = {}
        
        for col in cluster_df.columns:
            avrg = np.mean(cluster_df[col])
            district_id = int(col.split('_')[1])
            
            avrg_profit[district_id] = avrg
            
        dtag_gdf['cluster_'+str(cluster)] = dtag_gdf['DN'].map(avrg_profit)
    return dtag_gdf
    
def calculate_e_cluster(df, cluster_col, cols_to_delete=None):
    df_ = df.copy()
    clusters = df_[cluster_col]
    del df_[cluster_col]
    
    if cols_to_delete:
        for col in cols_to_delete:
            del df_[col]
            
    all_errors = []
    for col in df_.columns:
        means = [np.mean(df_.loc[clusters==x][col]) for x in np.unique(clusters)] #take the mean of each cluster
        sum_squared_errors = [(df_.loc[clusters==x][col]-means[i])**2 for i,x in enumerate(np.unique(clusters))]
        sum_squared_errors = [np.sum(x) for x in sum_squared_errors]
        all_errors.append(np.sum(sum_squared_errors))
        
    return np.sum(all_errors)

def calculate_e_dataset(df, cols_to_delete=None):
    df_ = df.copy()
    
    if cols_to_delete:
        for col in cols_to_delete:
            del df_[col]
            
    all_errors = []
    for col in df_.columns:
        mean = np.mean(df_[col]) #take the mean of each cluster
        sum_squared_errors = (df_[col]-mean)**2 
        all_errors.append(np.sum(sum_squared_errors))
        
    return np.sum(all_errors)

def evaluate_explained_variance(n_cluster, X):
    df_ = X.copy()
    EV_dict = {}
    EV_increase_dict = {}
    
    #GMM
    REs = []
    RE_increase = []
    print('start GMM')
    for i in range(n_cluster):
        print(i, sep=' ', end='', flush=True)
        clusterer = GMM(n_components=i+2).fit(X)
        labels_used = clusterer.predict(X)
        df_['labels'] = labels_used

        REs.append(1-calculate_e_cluster(df_, 'labels')/calculate_e_dataset(df_, cols_to_delete=['labels']))

        if i > 0:
            RE_increase.append(REs[i]-REs[i-1])
        else:
            RE_increase.append(REs[i])
    
    EV_dict['GMM'] = REs
    EV_increase_dict['GMM'] = RE_increase
    
    #kmedoids
    REs = []
    RE_increase = []
    print('start kmedoids')
    for i in range(n_cluster):
        print(i, sep=' ', end='', flush=True)
        clusterer = KMedoids(n_clusters=i+2, random_state=10, init='k-medoids++', metric='euclidean')
        labels_used = clusterer.fit_predict(X)
        df_['labels'] = labels_used

        REs.append(1-calculate_e_cluster(df_, 'labels')/calculate_e_dataset(df_, cols_to_delete=['labels']))

        if i > 0:
            RE_increase.append(REs[i]-REs[i-1])
        else:
            RE_increase.append(REs[i])
    
    EV_dict['kmedoids'] = REs
    EV_increase_dict['kmedoids'] = RE_increase
    
    #kmeans
    REs = []
    RE_increase = []
    print('start kmeans')
    for i in range(n_cluster):
        print(i, sep=' ', end='', flush=True)
        clusterer = KMeans(n_clusters=i+2, random_state=10)
        labels_used = clusterer.fit_predict(X)
        df_['labels'] = labels_used

        REs.append(1-calculate_e_cluster(df_, 'labels')/calculate_e_dataset(df_, cols_to_delete=['labels']))

        if i > 0:
            RE_increase.append(REs[i]-REs[i-1])
        else:
            RE_increase.append(REs[i])
    
    EV_dict['kmeans'] = REs
    EV_increase_dict['kmeans'] = RE_increase
    
    #agg-averarge
    REs = []
    RE_increase = []
    print('start agg-average')
    for i in range(n_cluster):
        print(i, sep=' ', end='', flush=True)
        clusterer = AgglomerativeClustering(n_clusters=i+2, linkage='average')
        labels_used = clusterer.fit_predict(X)
        df_['labels'] = labels_used

        REs.append(1-calculate_e_cluster(df_, 'labels')/calculate_e_dataset(df_, cols_to_delete=['labels']))

        if i > 0:
            RE_increase.append(REs[i]-REs[i-1])
        else:
            RE_increase.append(REs[i])
    
    EV_dict['agg-average'] = REs
    EV_increase_dict['agg-average'] = RE_increase
    
    #agg-complete
    REs = []
    RE_increase = []
    print('start agg-complete')
    for i in range(n_cluster):
        print(i, sep=' ', end='', flush=True)
        clusterer = AgglomerativeClustering(n_clusters=i+2, linkage='complete')
        labels_used = clusterer.fit_predict(X)
        df_['labels'] = labels_used

        REs.append(1-calculate_e_cluster(df_, 'labels')/calculate_e_dataset(df_, cols_to_delete=['labels']))

        if i > 0:
            RE_increase.append(REs[i]-REs[i-1])
        else:
            RE_increase.append(REs[i])
    
    EV_dict['agg-complete'] = REs
    EV_increase_dict['agg-complete'] = RE_increase
    
    return EV_dict, EV_increase_dict

def plot_explained_variance(EV_dict, EV_increase_dict, vertical=False, thresval=0.05):
    x = [i+2 for i in range(len(EV_dict['kmeans']))]
    
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    
    i=0
    j=0
    for n, method in enumerate(EV_dict.keys()):
        ax1 = axes[i,j]
        
        color = 'tab:red'
        ax1.set_xlabel('n_components')
        ax1.set_ylabel('Explained variance', color=color)
        ax1.plot(x, EV_dict[method], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Delta explained variance', color=color)  # we already handled the x-label with ax1
        ax2.plot(x, EV_increase_dict[method], color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=thresval, color='grey', linestyle='--')
        index_threshold = EV_increase_dict[method].index(next(i for i in EV_increase_dict[method] if i < thresval))
        if vertical:
            ax2.axvline(x=index_threshold+2,color='grey', linestyle='--')
        
        ax1.set_ylim(0.1, 1)
        ax2.set_ylim(0, 0.4)
        ax1.set_title(method)
        
        print(method, EV_dict[method][index_threshold], index_threshold+2)
        
        if i < 1:
            i += 1
        else:
            j += 1
            i = 0
            
    plt.tight_layout()
    fig.delaxes(axes[-1][-1])