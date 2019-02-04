from PreProcessing import PreProcessing
import numpy as np
import xml.etree.ElementTree as ET
from nltk.stem.wordnet import WordNetLemmatizer


def point_wise_add(x, y):
    result = []
    assert len(x) == len(y)
    for idx in range(len(x)):
        result.append(x[idx] + y[idx])
    return result


def point_wise_compare(x, y):
    # x <= y
    assert len(x) == len(y)
    for idx in range(len(y)):
        if x[idx] > y[idx]:
            return False
    return True


def scaler_vector_mult(s, v):
    result = []
    for idx in range(len(v)):
        result.append(int(s * v[idx]))
    return result


class LoadDataset():
    def __init__(self, redundant=False):
        self.category_label_num = {
            'service': 0,
            'food': 1,
            'price': 2,
            'ambience': 3,
            'anecdotes/miscellaneous': 4
        }
        self.percentage = 1.0
        self.redundant = redundant
        self.extract_data('dataset/ABSA14_Restaurants_Train.xml',
                                     'dataset/Restaurants_Test_Data_phaseB.xml',)

    def __len__(self):
        return len(self.train_data)

    def get_sentences(self, cat):
        result = []
        for data_instance in self.train_data:
            # if data_instance[1][self.category_label_num[cat]] != 1:
            #     continue
            sentence = ' '.join(data_instance[0])
            for word in sentence:
                if word.isdigit():
                    sentence = sentence.replace(word, ' ')
            sentence = sentence.split()
            result.append(sentence)
        return result

    def get_sentences_by_partial_label(self, p_label):
        result = []
        for data_instance in self.train_data:
            if p_label.lower() not in data_instance[1].lower():
                continue
            sentence = ' '.join(data_instance[0])
            for word in sentence:
                if word.isdigit():
                    sentence = sentence.replace(word, ' ')
            sentence = sentence.split()
            if sentence not in result:
                result.append(sentence)
        return result

    def get_labels(self, cat):
        result = []
        for data_instance in self.train_data:
            if data_instance[1][self.category_label_num[cat]] == 1:
                result.append(1)
            else:
                result.append(0)
        return result

    def extract_data(self, train_file, test_file):
        tree = ET.parse(train_file)
        root = tree.getroot()
        train_sentences = root.findall('sentence')
        tree = ET.parse(test_file)
        root = tree.getroot()
        test_sentences = root.findall('sentence')
        self.train_sentence_with_all_labels = {}
        self.train_labels = {}
        self.test_labels = {}
        self.processed_train_sentences = self.process_data(train_sentences)
        self.processed_test_sentences = self.process_data(test_sentences)
        self.original_train_sentences = self.getOriginalsentences(train_sentences)
        self.original_test_sentences = self.getOriginalTestsentences(test_sentences)
        self.train_data, self.categories = self.get_inputs(self.processed_train_sentences,
                                                           train_sentences,
                                                           is_train=True)
        self.test_data = self.get_inputs(self.processed_test_sentences,
                                         test_sentences)
        self.number_of_categories = len(self.categories)
        print(self.categories)
        print(len(self.train_data))

    def getOriginalsentences(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)
        return unprocessed_sentences

    def getOriginalTestsentences(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)
        return unprocessed_sentences

    def process_data(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)
        preprocessor = PreProcessing(unprocessed_sentences, 'english')
        preprocessor.Remove_Punctuation()
        processed_sentences = preprocessor.Remove_StopWords()

        return processed_sentences

    def get_inputs(self, processed_sentences, unprocessed_data, is_train=False):
        processed_data = []
        categories = []
        lmtz = WordNetLemmatizer()
        length_1 = 0
        length_2 = 0
        length_3 = 0
        length_4 = 0
        length_5 = 0
        for i in range(len(processed_sentences)):
            processed_sentences[i] = processed_sentences[i].split()
        for i in range(len(unprocessed_data)):
            sentence = processed_sentences[i]
            # sentence = [lmtz.lemmatize(word) for word in sentence]
            sentence_categories = []
            if len(unprocessed_data[i]) > 1 and len(unprocessed_data[i][1]) > 0:
                if unprocessed_data[i][1].tag == 'aspectCategories':
                    aspect_cats = unprocessed_data[i][1]
                else:
                    aspect_cats = unprocessed_data[i][2]
                if is_train:
                    labels = 5 * [0]
                    for opinions in aspect_cats:
                        dict = opinions.attrib
                        if dict['category'] in sentence_categories:
                            continue
                        labels[self.category_label_num[str(dict['category'])]] = 1
                        sentence_categories.append(dict['category'])
                        if dict['category'] not in categories:
                            categories.append(dict['category'])
                    processed_data.append([sentence, labels])
                    self.train_sentence_with_all_labels[' '.join(sentence)] = labels
                    self.train_labels[i] = sentence_categories
                else:
                    test_sentence_categories = []
                    # sentence_categories = []
                    if unprocessed_data[i][1].tag == 'aspectCategories':
                        aspect_cats = unprocessed_data[i][1]
                    else:
                        aspect_cats = unprocessed_data[i][2]
                    for opinions in aspect_cats:
                        dict = opinions.attrib
                        if self.category_label_num[dict['category']] not in test_sentence_categories:
                            test_sentence_categories.append(self.category_label_num[dict['category']])
                            sentence_categories.append(dict['category'])
                    processed_data.append([sentence, test_sentence_categories])
                    self.test_labels[i] = sentence_categories
                    if 0 in test_sentence_categories:
                        length_1 += 1
                    if 1 in test_sentence_categories:
                        length_2 += 1
                    if 2 in test_sentence_categories:
                        length_3 += 1
                    if 3 in test_sentence_categories:
                        length_4 += 1
                    if 4 in test_sentence_categories:
                        length_5 += 1
            else:
                if is_train:
                    processed_data.append([sentence, 'NULL'])
                    self.train_sentence_with_all_labels[' '.join(sentence)] = 5 * [0]
                    self.train_sentence_with_all_labels[' '.join(sentence)][-1] = 1
                else:
                    processed_data.append([sentence, [self.category_label_num['NULL']]])
        if is_train:
            num_of_inst_for_each_label = len(self.category_label_num.keys()) * [0]
            for instance in processed_data:
                num_of_inst_for_each_label = point_wise_add(num_of_inst_for_each_label, instance[1])

            temp_data_container = []
            temp_num_of_inst_for_each_label = len(self.category_label_num.keys()) * [0]
            for instance in processed_data:
                if point_wise_compare(point_wise_add(instance[1], temp_num_of_inst_for_each_label),
                                      scaler_vector_mult(self.percentage, num_of_inst_for_each_label)):
                    temp_data_container.append(instance)
                    temp_num_of_inst_for_each_label = point_wise_add(instance[1], temp_num_of_inst_for_each_label)
            print(temp_num_of_inst_for_each_label)
            processed_data = temp_data_container
            if self.redundant:
                temp_data_container = []
                for instance in processed_data:
                    for idx in range(len(instance[1])):
                        if instance[1][idx] == 1:
                            label = len(self.category_label_num.keys()) * [0]
                            label[idx] = 1
                            temp_data_container.append([instance[0], label])
                processed_data = temp_data_container
        if is_train:
            return processed_data, categories
        else:
            return processed_data


# if __name__ == '__main__':
#     dataset = dataset = SimpleDataset('../../datas/ABSA14_Restaurants_Train.xml',
#                                       '../../datas/Restaurants_Test_Data_phaseB.xml', 0.1, redundant=True)