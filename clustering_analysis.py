import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

exclude_zeros = True
scale = True
method = 'PCA'
density_plot = False
kmeans_clustering_test = False
kmeans_clustring = True
plot_scatter_clustring = True
plot_summery_clustring = False
cmeans_clustering_test = False
cmeans_clustring = False
number_clusters = 5


df = pd.read_csv("chip_6histon_varients_ave100kb.txt",sep="\t")
data0 = df.drop(['chromosome','start','end'], 1)
if (exclude_zeros):
    data = data0.loc[(data0!=0).any(axis=1)]
else:
    data = data0

if (scale):
    from sklearn.preprocessing import StandardScaler
    print('scaling data started')
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    print('scaling data finished')

if (method=='PCA'):
    from sklearn.decomposition import PCA
    print('PCA computation started')
    pca = PCA(n_components=data.shape[1])
    if (scale):
        pca.fit(scaled_data)
        reduced_data = pca.fit_transform(scaled_data)
    else:
        pca.fit(data)
        reduced_data = pca.fit_transform(data)
    variances = pca.explained_variance_ratio_
    per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
    labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]
    print('PCA computation finished')
elif(method=='TSNE'):
    from sklearn.manifold import TSNE
    print('t-SNE computation started')
    if scale:
        reduced_data = TSNE(n_components=2).fit_transform(scaled_data)
    else:
        reduced_data = TSNE(n_components=2).fit_transform(data)
    print('t-SNE computation finished')

x = reduced_data[:,0]
y = reduced_data[:,1]
combined_data = np.vstack((x, y)).T

if (density_plot):
    from scipy.stats import gaussian_kde
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    cc = cm.jet((z-z.min())/(z.max()-z.min()))
    plt.scatter(x, y, c=cc, s=2)
    plt.set_cmap('jet')                                                                        
    plt.colorbar()
    if (method=='PCA'):
        plt.xlabel('PC1:{0}%'.format(per_var[0]))
        plt.ylabel('PC2:{0}%'.format(per_var[1]))
    elif(method=='TSNE'):
        plt.xlabel('t-SNE1')                                                                           
        plt.ylabel('t-SNE2')   


if (kmeans_clustering_test):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    a = np.zeros((9, 3))
    for number_clusters in range(1,10):
        kmeans = KMeans(init="k-means++", n_clusters=number_clusters+1, n_init=10)
        kmeans.fit(combined_data)
        Z = kmeans.fit_predict(combined_data)
        a[number_clusters-1,0] = number_clusters + 1
        a[number_clusters-1,1] = silhouette_score(combined_data, Z, metric="euclidean")                                 
        a[number_clusters-1,2] = -kmeans.score(combined_data)
        print(number_clusters+1)
    plt.subplot(2,1,1)
    plt.plot(a[:,0],a[:,2])
    plt.xlabel('')
    plt.ylabel('k-means score')
    plt.subplot(2,1,2)
    plt.plot(a[:,0],a[:,1])
    plt.xlabel('# clusters')
    plt.ylabel('silhouette score')

if (kmeans_clustring):
    from sklearn.cluster import KMeans
    kmeans = KMeans(init="k-means++", n_clusters=number_clusters, n_init=20)
    kmeans.fit(combined_data)
    Z = kmeans.predict(combined_data)
    if (plot_summery_clustring):
        import collections
        elements_count = collections.Counter(Z)
        count = 0
        per_var =np.zeros(len(elements_count))
        i=0
        for key, value in elements_count.items():
            per_var[i] = value
            count += value
            i +=1
        plt.bar(x=range(1,len(per_var)+1),height=100*per_var/count)
        plt.xlabel('state')
        plt.ylabel('%')
    if (plot_scatter_clustring):
        plt.scatter(x, y, c=Z, s=2)
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=16, linewidths=3,color="r", zorder=10)
        if (method=='PCA'):
            plt.xlabel('PC1:{0}%'.format(per_var[0]))
            plt.ylabel('PC2:{0}%'.format(per_var[1]))
            plt.title('PCA based clustering, #cluster='+str(number_clusters))
        elif(method=='TSNE'):
            plt.xlabel('t-SNE1')
            plt.ylabel('t-SNE2')
            plt.title('t-SNE based clustering, #cluster='+str(number_clusters))

if (cmeans_clustering_test):
    import skfuzzy as fuzz
    fpcs = []
    for number_clusters in range(2,11):
        cntr, _, _, _, _, _, fpc = fuzz.cluster.cmeans(combined_data.T, number_clusters, 2, error=0.005, maxiter=1000, init=None)
        fpcs.append(fpc)
    plt.plot(np.r_[2:11], fpcs)
    plt.xlabel('# clusters')
    plt.ylabel('Fuzzy partition coefficient')

if(cmeans_clustring):
    import skfuzzy as fuzz
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(combined_data.T, number_clusters, 2, error=0.005, maxiter=1000)
    if (plot_scatter_clustring):
        cluster_membership = np.argmax(u, axis=0)
        plt.scatter(x, y, c=cluster_membership, s=2)
        for pt in cntr:
            plt.plot(pt[0], pt[1], 'rs')
        if (method=='PCA'):
            plt.xlabel('PC1:{0}%'.format(per_var[0]))
            plt.ylabel('PC2:{0}%'.format(per_var[1]))
            plt.title('PCA based fuzzy clustering, #cluster='+str(number_clusters))
        elif(method=='TSNE'):
            plt.xlabel('t-SNE1')
            plt.ylabel('t-SNE2')
            plt.title('t-SNE based fuzzy clustering, #cluster='+str(number_clusters))
    

plt.show()
