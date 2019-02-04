from gensim import models
from dataLoader import LoadDataset
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from gensim.matutils import softcossim
from gensim import corpora
from copy import deepcopy

def sigmoid(x, derivative=False):
    return x * (1-x) if derivative else 1 / (1+np.exp(-x))



class Unsupervised:
    def __init__(self):
        self.w2v_model = models.KeyedVectors.load_word2vec_format('word-embedding/yelp_W2V_300_orig.bin', binary=True)
        self.category_label_num = {
            'service': 0,
            'food': 1,
            'price': 2,
            'ambience': 3,
            'anecdotes/miscellaneous': 4
        }
        self.category_num_label = {
            0: 'service',
            1: 'food',
            2: 'price',
            3: 'ambience',
            4: 'anecdotes/miscellaneous'
        }
        self.category_seed_words = {
            'service': {'service', 'staff', 'friendly', 'attentive', 'manager'},
            'food': {'food', 'delicious', 'menu', 'fresh', 'tasty'},
            'price': {'price', 'cheap', 'expensive', 'money', 'affordable'},
            'ambience': {'ambience', 'atmosphere', 'decor', 'romantic', 'loud'}
        }
        self.categories = [
            'service',
            'food',
            'price',
            'ambience',
            'anecdotes/miscellaneous'
        ]
        self.dataset = LoadDataset()
        self.yelp_sentences = []
        self.getYelpSentences()
        self.test_sentences = []
        self.test_sentences_with_label = []
        for item in self.dataset.test_data:
            self.test_sentences.append(item[0])
            self.test_sentences_with_label.append(item)

        self.dictionary = corpora.Dictionary(self.test_sentences)
        self.corpus = [self.dictionary.doc2bow(document) for document in self.test_sentences]
        self.similarity_matrix = self.w2v_model.similarity_matrix(self.dictionary)
        # np.save("similarity_matrix", self.similarity_matrix)
        # self.similarity_matrix = np.load("similarity_matrix.npy").item()

    def getYelpSentences(self):
        f = open('yelp-weak-supervision/yelp_restaurant_review.txt', 'r')
        raw_yelp = f.read().split('\n')
        yelp_sentences = [raw_yelp[i].split(' ') for i in range(len(raw_yelp))]
        random_samples = random.sample(yelp_sentences, 10000)
        self.yelp_sentences = random_samples

    def sentence_embedd_average(self, sentence):
        sum_of_words = np.zeros(300)
        for word in sentence:
            try:
                sum_of_words += self.w2v_model[word]
            except KeyError:
                continue
        return sum_of_words / len(sentence)

    def get_distances(self, center, points):
        distances = []
        for point in points:
            distances.append(np.linalg.norm(point - center))
        return distances

    def initialize_clusters(self, points, k):
        random_points = []
        while True:
            random_num = np.random.randint(0, len(points))
            random_points.append(self.sentence_embedd_average(points[random_num]))
            if len(random_points) == k:
                break
        return random_points

    def k_means_clustering_yelp(self, k):
        max_iter = 100
        centroids = self.initialize_clusters(self.yelp_sentences, k)

        cluster_indexs = len(self.yelp_sentences) * [0]
        old_loss = float('inf')
        print('Clustering yelp sentences into ' + str(k) + ' clusters.')
        for i in range(max_iter):
            distances = []
            loss = 0
            for j in range(len(self.yelp_sentences)):
                centroid_distances = self.get_distances(
                    self.sentence_embedd_average(self.yelp_sentences[j]), centroids)
                distances.append(centroid_distances)
            for j in range(len(distances)):
                try:
                    cluster_indexs[j] = int(np.argmin(distances[j]))
                    loss += distances[j][cluster_indexs[j]]
                except:
                    print(distances[j])
            if loss >= old_loss:
                print('Converged in ' + str(i) + ' iterations.')
                break
            else:
                old_loss = loss
            for c in range(k):
                if len([j for j in range(len(self.yelp_sentences)) if cluster_indexs[j] == c]) < 2:
                    continue
                centroids[c] = np.mean([self.sentence_embedd_average(self.yelp_sentences[j])
                                        for j in range(len(self.yelp_sentences))
                                        if cluster_indexs[j] == c], 0)
        return cluster_indexs, centroids

    def get_category_seed_similarity(self, sentence, seeds, similarity_matrix):
        result = 0
        length = len(seeds)
        sentence_d2b = self.dictionary.doc2bow(sentence)
        for word in seeds:
            seed_d2b = self.dictionary.doc2bow([word])
            result += softcossim(sentence_d2b, seed_d2b, similarity_matrix)
        return result / length

    def classify_clusters(self, cluster_indexes, centroids):
        clustScore = defaultdict(list)
        print('Classifying clusters.')
        for c in tqdm(range(len(centroids))):
            cluster_sentences = [self.yelp_sentences[j] for j in range(len(self.yelp_sentences))
                                 if cluster_indexes[j] == c]
            category_similarities = len(self.categories) * [0.0]
            for sentence in cluster_sentences:
                for cat in self.categories:
                    if cat == 'anecdotes/miscellaneous':
                        continue
                    seeds = self.category_seed_words[cat]
                    category_similarities[self.category_label_num[cat]] += \
                        sigmoid(self.get_category_seed_similarity(sentence, seeds,
                                                                  self.similarity_matrix))
            try:
                category_similarities = [category_similarities[j] / len(cluster_sentences)
                                         for j in range(len(category_similarities))]
            except ZeroDivisionError:
                pass
            clustScore[c] = category_similarities[:]
            print(clustScore)
        return clustScore

    def getClusterScores(self, sentence, centroids, cluster_category_similarities):
        centroid_distances = self.get_distances(self.sentence_embedd_average(sentence), centroids)
        nearest_cluster = int(np.argmin(centroid_distances))
        return cluster_category_similarities[nearest_cluster]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def cos_sim(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def find_best_threshold(self, predicted_scores, gtruth_labels):
        threshold = 0.19
        best_res = [0.0, 0.0, 0.0, 0.0]
        while threshold < 0.7:
            TP = 0
            FP = 0
            FN = 0
            for i in range(len(gtruth_labels)):
                pred_labels = deepcopy(predicted_scores[i])
                ground_truth = deepcopy(gtruth_labels[i])
                for idx in range(len(pred_labels)):
                    if pred_labels[idx] > threshold:
                        pred_labels[idx] = 1
                    else:
                        pred_labels[idx] = 0
                if np.any(pred_labels) == 0:
                    pred_labels[4] = 1
                for idx in range(len(pred_labels)):
                    if pred_labels[idx] == 1:
                        if idx in ground_truth:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if idx in ground_truth:
                            FN += 1
            try:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.0
            if f1 > best_res[0]:
                best_res = [f1, precision, recall, threshold]
            threshold += 0.0001
            threshold = round(threshold, 4)
        return best_res

    def classify_test_sentences(self, alpha, cluster_category_similarities, centroids):
        predicted_scores = []
        gtruth_labels = []
        print('Classifying test sentences ...')
        for i, sentence in tqdm(enumerate(self.test_sentences)):
            clustScore = self.getClusterScores(sentence, centroids, cluster_category_similarities)
            gtruth_label = self.test_sentences_with_label[i][1]

            sentScore = len(self.categories) * [0.0]
            for cat in self.categories:
                if cat == 'anecdotes/miscellaneous':
                    continue
                seeds = self.category_seed_words[cat]
                sentScore[self.category_label_num[cat]] += sigmoid(
                    self.get_category_seed_similarity(sentence, seeds, self.similarity_matrix))

            sentScore_N = np.array(sentScore) / np.linalg.norm(np.array(sentScore))
            clustScore_N = np.array(clustScore) / np.linalg.norm(np.array(clustScore))
            score = alpha * sentScore_N + (1-alpha) * clustScore_N

            predicted_scores.append(deepcopy(score))
            gtruth_labels.append(deepcopy(gtruth_label))
        result = self.find_best_threshold(predicted_scores, gtruth_labels)
        return result
