from model import Unsupervised






def main():
    num_clusters = 12
    alpha = 0.7
    unsupervised = Unsupervised()
    cluster_indexes, centroids = unsupervised.k_means_clustering_yelp(num_clusters)
    clustScores = unsupervised.classify_clusters(cluster_indexes, centroids)
    result = unsupervised.classify_test_sentences(alpha, clustScores, centroids)
    print('F1-measure : ' + str(result[0]))
    print('Precision : ' + str(result[1]))
    print('Recall : ' + str(result[2]))
    print('threshold : ' + str(result[3]))

if __name__ == '__main__':
    main()
