from natasha import MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger, Doc
from data_import import *
import json
from collections import Counter
import numpy as np

class TextRank:

    def __init__(self, idf_path=None, mode="tf-idf", setted_idf=False):
        self.morph_vocab = MorphVocab()
        self.segmenter = Segmenter()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.stop_words = []
        self.setted_idf = setted_idf

        self.mode = mode

        self.load_stop_words('stopwords-ru.txt')
        self.idf = {}

        if idf_path is not None:
            self.load_idf(idf_path)

    def load_idf(self, filepath="idf.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.idf = json.load(f)

    def load_stop_words(self, file):
        self.stop_words = []

        with open(file, mode='r', encoding='utf-8') as words_f:
            for line in words_f:
                self.stop_words.append(line.strip("\n"))

    def lemmatize(self, tokens):

        for token in tokens:
            token.lemmatize(self.morph_vocab)

        lemmas = [word.lemma for word in tokens]
        return lemmas

    def save_idf(self, filepath="idf.json"):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.idf, f, ensure_ascii=False, indent=4)

    def save_data(self, data, filepath="data.json"):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def compute_idf(self, token, texts):
        N = len(texts)
        cont = 0
        for text in texts:
            if token in text:
                cont += 1
        idf = np.log((1 + N) / (1 + cont)) + 1

        return idf

    def get_idf(self, token, texts, idfs):
        if token not in idfs.keys():
            return self.compute_idf(token, texts)
        else:
            return idfs[token]

    def set_idf(self, texts):

        N = len(texts)
        conts = {}
        cont = 0
        for paper in texts:
            cont += 1
            print(cont)
            doc = Doc(paper)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)

            lemmas = set()
            for token in doc.tokens:

                if token.pos == "PUNCT":
                    continue

                token.lemmatize(self.morph_vocab)
                lemmas.add(token.lemma)


            for lemma in lemmas:
                if lemma in conts.keys():
                    conts[lemma] += 1
                else:
                    conts[lemma] = 1

        for token, count in conts.items():
            idf = np.log((1 + N) / (1 + count)) + 1
            self.idf[token] = idf

    def del_stop_words(self, tokens):
        words_indexes = []

        for i, token in enumerate(tokens):
            if not (token.text in self.stop_words):
                words_indexes.append(i)

        return words_indexes

    def del_punc(self, tokens):
        punc_indexes = []

        for i, token in enumerate(tokens):
            if token.pos != "PUNCT":
                punc_indexes.append(i)

        return punc_indexes

    def get_summary_mass(self, texts, limit):

        res = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processed {i} texts...")
            self.get_summary(text, limit)
            res.append(self.get_summary(text, limit))

        return res

    def get_summary(self, text, limit=3, data_save=False):

        data = self.prepare(text)
        if data_save:
            self.save_data(data)
        sents = data["lemmas_no_punc_stop"]
 
        sim_matrix = None
        if self.mode == "tf-idf":
            sim_matrix = self.get_similarity_matrix_tf_idf(sents)
        else:
            sim_matrix = self.get_similarity_matrix(sents)
        scores = self.get_rank(sim_matrix, 0.85)

        scores_sort = np.argsort(scores)
        scores_sort = scores_sort[-limit:]
        scores_sort = np.sort(scores_sort)

        needed_clusters = None

        needed_clusters = np.array(data["sentences"])[scores_sort].tolist()
        
        return needed_clusters

    def get_rank(self, M, d, number_iter=100):

        n = M.shape[0]
        sum_by_rows = M.sum(axis=0)
        sum_by_rows[sum_by_rows == 0] = 1
        eval_matrix = M / sum_by_rows * d

        minus_coeff = (1 - d) / n

        r = np.array([1] * n, dtype=float)

        while number_iter:
            x = eval_matrix @ r + minus_coeff
            r = x / np.sum(x)

            number_iter -= 1

        return r

    def get_similarity_matrix(self, sentences):

        n = len(sentences)
        sim_array = np.zeros((n, n))

        for i, s1 in enumerate(sentences):
            for j, s2 in enumerate(sentences):

                if (i == j):
                    sim_array[i][j] = 0
                    continue

                common_words_count = len(set(s1).intersection(set(s2)))
                s1_count = len(s1)
                s2_count = len(s2)

                sim_array[i][j] = common_words_count / (np.log(s1_count + 1) + np.log(s2_count + 1))
        return sim_array

    def get_similarity_matrix_tf_idf(self, sentences):
    
        n = len(sentences)
        sim_array = np.zeros((n, n))
        tfs = []
        N = len(sentences)

        for i, s1 in enumerate(sentences):

            tf = dict(Counter(s1))
            tfs.append(tf)

            if not self.setted_idf:
                self.idf = {}
                for token in s1:
                    isin = np.array([1 if token in sen else 0 for sen in sentences])
                    idf = np.log((1 + N) / (1 + np.sum(isin))) + 1
                    self.idf[token] = idf

        for j, s1 in enumerate(sentences):
            for k, s2 in enumerate(sentences):
                if (k == j):
                    sim_array[i][j] = 0
                    continue

                common_words = set(set(s1).intersection(set(s2)))
                sum_words = np.sum(np.array([tfs[j][word] * tfs[k][word] * self.get_idf(word, sentences, self.idf)**2 for word in common_words]))
                s1_sum = np.sqrt(np.sum(np.array([(tfs[j][word] * self.get_idf(word, sentences, self.idf))**2 for word in set(s1)])))
                s2_sum = np.sqrt(np.sum(np.array([(tfs[k][word] * self.get_idf(word, sentences, self.idf))**2 for word in set(s2)])))

                denom = (s1_sum * s2_sum)
                if denom:
                    sim_array[j][k] = sum_words / (s1_sum * s2_sum)
                else:
                    sim_array[j][k] = 0
                    
        return sim_array

    def prepare(self, text):

        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        sents = []
        words = []
        lemmas = []
        lemmas_no_punc_stop = []
        global_lemmas = []
        stop = []
        punc = []

        for sent in doc.sents:
            sents.append(sent.text)
            words.append([token.text for token in sent.tokens])
            current_lems = self.lemmatize(sent.tokens)
            lemmas.append(current_lems)

            current_stop = self.del_stop_words(sent.tokens)
            current_punc = self.del_punc(sent.tokens)
            n_words = list(set(current_stop).intersection(set(current_punc)))
            n_words = sorted(n_words)
            punc.append(current_punc)
            stop.append(current_stop)

            lemmas_no_punc_stop.append(np.array(current_lems)[n_words].tolist())
            global_lemmas += current_lems

        return {
                "words": words, 
                "sentences": sents, 
                "lemmas_no_punc_stop": lemmas_no_punc_stop,
                "lemmas": lemmas, 
                "global_lemmas": global_lemmas, 
                "no_stop_words": stop, 
                "no_punc_words": punc
                }