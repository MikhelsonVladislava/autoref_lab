from natasha import MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger, Doc
from collections import Counter
import numpy as np

class Luhn:

    def __init__(self, mode="parts", score_method="count", ind_coeff=6):
        self.ind_coeff = ind_coeff
        self.morph_vocab = MorphVocab()
        self.segmenter = Segmenter()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.stop_words = []

        self.score_method = score_method

        self.mode = mode

        self.load_stop_words('stopwords-ru.txt')

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

    def get_sentences_classes(self, words, punc, stop, lemmas, freq_all, upper_border, lower_border):
        
        clusters = []
        scores = []
        freq = None

        for i, sentences in enumerate(words):
            
            if self.score_method == "count":
                freq = freq_all[0]
            else:
                freq = freq_all[i]
                
            clusters_current = []
            needed_words_current = []
            indexes = list(set(punc[i]).intersection(set(stop[i])))
            for j, lemma in enumerate(lemmas[i]):
                if lemma in freq.keys() and freq[lemma] <= upper_border and freq[lemma] >= lower_border and j in indexes:
                    needed_words_current.append(j)

            prev_index = 0
            cluster = []
            for k, index in enumerate(needed_words_current):
                if k == 0:
                    prev_index = index
                    cluster.append(prev_index)
                    continue

                if index - prev_index <= self.ind_coeff:
                    cluster.append(index)
                else:
                    if len(cluster) > 1:
                        clusters_current.append(cluster)

                    cluster = []
                
                prev_index = index

                if (k == (len(needed_words_current) - 1) and len(cluster) > 1):
                    clusters_current.append(cluster)

            cum_cluster = []
            max_score = 0
            for cls in clusters_current:
                start = cls[0]
                end = cls[-1]
                score = len(cls)**2 / (end - start)
                if max_score < score:
                    max_score = score
                
                cum_cluster += sentences[cls[0]:cls[-1]]

            scores.append(max_score)
            clusters.append(" ".join(cum_cluster) + '.')

        return clusters, scores

    def get_summary_mass(self, texts, limit):

        res = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processed {i} texts...")
            res.append(self.get_summary(text, limit))

        return res

    def get_summary(self, text, limit):

        data = self.prepare(text)
        clusters, scores = self.get_sentences_classes(data["words"], 
                                                      data["no_punc_words"], 
                                                      data["no_stop_words"], 
                                                      data["lemmas"], 
                                                      data["freq"], 
                                                      data["upper_border"], 
                                                      data["lower_border"])

        scores_array = np.array(scores)
        scores_sort = np.argsort(scores_array)
        scores_sort = scores_sort[-limit:]
        scores_sort = np.sort(scores_sort)

        needed_clusters = None

        if self.mode == "parts":
            needed_clusters = np.array(clusters)[scores_sort].tolist()
        else:
            needed_clusters = np.array(data["sentences"])[scores_sort].tolist()
        return needed_clusters

    def tf_idf(self, sentences):

        tfs = []
        scores = []
        scores_list = []
        for sentence in sentences:
            tfs.append(Counter(sentence))

        for i, sentence in enumerate(sentences):
            idf = 0
            lens = len(sentence)
            current_scores = {}
            current_scores_list = []
            for token in sentence:

                isin = np.array([1 if token in sen else 0 for sen in sentences])
                tf = tfs[i][token]
                idf = np.sum(isin)

                score = np.log(1 + tf) * np.log(lens / idf)
                current_scores[token] = score
                current_scores_list.append(score)
            
            scores.append(current_scores)
            scores_list.append(current_scores_list)
        return scores, scores_list

    def prepare(self, text):

        # text = text.lower().strip()

        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        sents = []
        words = []
        lemmas = []
        lemmas_no_punc = []
        global_lemmas = []
        global_lemmas_no_punc = []
        stop = []
        punc = []

        for sent in doc.sents:
            sents.append(sent.text)
            words.append([token.text for token in sent.tokens])
            current_lems = self.lemmatize(sent.tokens)
            lemmas.append(current_lems)

            current_stop = self.del_stop_words(sent.tokens)
            current_punc = self.del_punc(sent.tokens)
            punc.append(current_punc)
            stop.append(current_stop)

            lemmas_no_punc.append(np.array(current_lems)[current_punc].tolist())
            global_lemmas += current_lems
            global_lemmas_no_punc += np.array(current_lems)[current_punc].tolist()

        freq = None
        lower_border = None
        upper_border = None
        if self.score_method == "count":
            freq = dict(Counter(global_lemmas_no_punc))
            stop_words_freq = [freq[word] if word in freq.keys() else 0 for word in self.stop_words]
            stop_words_freq.sort()
            stop_words_freq = stop_words_freq[-10:]
            upper_border = stop_words_freq[0]
            lower_border = 1
            freq = [freq]
        else:
            freq, statistic_list = self.tf_idf(lemmas_no_punc)
            global_freq = np.concatenate([np.array(sub) for sub in statistic_list])
            perc = np.percentile(global_freq, 80)
            upper_border = global_freq.max()
            lower_border = perc

        return {
                "words": words, 
                "sentences": sents, 
                "lemmas": lemmas, 
                "global_lemmas": global_lemmas, 
                "no_stop_words": stop, 
                "no_punc_words": punc,
                "freq": freq,
                "upper_border": upper_border,
                "lower_border": lower_border
                }
