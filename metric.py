from lexrank import *
from luhn import *
from natasha import Doc
from collections import Counter

def firts_n(sentences, limit):

    result = []
    cum_len = 0
    last = 0

    for i, sent in enumerate(sentences):

        if i == 0 and len(sent) > limit:
            return [sent[:limit]]
        
        if len(sent) + cum_len < limit:
            cum_len += len(sent) + 1
            result.append(sent)
            last = i

    if cum_len < limit and last + 1 < len(sentences):
        reduce = limit - cum_len
        result.append(sentences[last + 1][:reduce])

    return result

def make_text(sentences):

    text = " ".join(sentences)
    return text

def get_ngramms(tokens, n):

    if len(tokens) <= n:
        return " ".join(tokens)

    ngm = []
    for i in range(n, len(tokens) + 1):
        cur = " ".join(tokens[i - n:i])
        ngm.append(cur)

    return ngm

def rouge(summary, gold_summary, segmenter, tagger, mode=1):
    doc_summary = Doc(summary)
    doc_summary.segment(segmenter)
    doc_summary.tag_morph(tagger)

    doc_gold_summary = Doc(gold_summary)
    doc_gold_summary.segment(segmenter)
    doc_gold_summary.tag_morph(tagger)

    summary_tokens = [token.text for token in doc_summary.tokens if token.pos != "PUNCT"]
    gold_summary_tokens = [token.text for token in doc_gold_summary.tokens if token.pos != "PUNCT"]

    summary_tokens = get_ngramms(summary_tokens, mode)
    gold_summary_tokens = get_ngramms(gold_summary_tokens, mode)

    summary_counter = dict(Counter(summary_tokens))
    gold_summary_counter = dict(Counter(gold_summary_tokens))

    intersection = 0
    for word in summary_counter.keys():

        if word in gold_summary_counter.keys():
            intersection += min(summary_counter[word], gold_summary_counter[word])

    pw = len(gold_summary_tokens)

    return intersection / pw


def rougef1(summary, gold_summary, segmenter, tagger, mode=1):
    doc_summary = Doc(summary)
    doc_summary.segment(segmenter)
    doc_summary.tag_morph(tagger)

    doc_gold_summary = Doc(gold_summary)
    doc_gold_summary.segment(segmenter)
    doc_gold_summary.tag_morph(tagger)

    summary_tokens = [token.text for token in doc_summary.tokens if token.pos != "PUNCT"]
    gold_summary_tokens = [token.text for token in doc_gold_summary.tokens if token.pos != "PUNCT"]

    summary_tokens = get_ngramms(summary_tokens, mode)
    gold_summary_tokens = get_ngramms(gold_summary_tokens, mode)

    summary_counter = dict(Counter(summary_tokens))
    gold_summary_counter = dict(Counter(gold_summary_tokens))

    intersection = 0
    for word in summary_counter.keys():

        if word in gold_summary_counter.keys():
            intersection += min(summary_counter[word], gold_summary_counter[word])

    pw = len(gold_summary_tokens)
    pw_cand = len(summary_tokens)

    recall = intersection / pw
    precision = intersection / pw_cand
    f1 = 0
    if precision + recall:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall
