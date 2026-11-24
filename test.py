from metric import *
import pickle
import matplotlib.pyplot as plt

train, test = get_train_test_gazeta('train.parquet', 'test.parquet')
text = test["text"].to_list()
gold = test["summary"].to_list()


segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
labels = []
rouge1_massive = []
rouge2_massive = []
rougef1_massive = []
rouge1f1_massive = []

luhn = Luhn(mode="sentences", score_method="count", ind_coeff=3)

result = luhn.get_summary_mass(text, 10)

N = len(result)
cum_rouge1 = cum_rouge2 = cumf1 = cum1f1 =  0
summaries_luhn = []
for i, res in enumerate(result):
    example = gold[i]
    t_r = make_text(firts_n(res, 300))
    summaries_luhn.append(t_r)

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 2)
    cum_rouge2 += pres
    cumf1 += f1

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 1)
    cum_rouge1 += pres
    cum1f1 += f1

with open('summaries_luhn.pkl', 'wb') as f:
    pickle.dump(list(zip(summaries_luhn, gold, text)), f)

rouge1_massive.append(cum_rouge1 / N)
rouge2_massive.append(cum_rouge2 / N)
rougef1_massive.append(cumf1 / N)
rouge1f1_massive.append(cum1f1 / N)
labels.append("Lhun (full)")
print(f"Lhun (full sentences) rouge1: {rouge1_massive[-1]}; rouge2: {rouge2_massive[-1]}; f1 rouge2: {rougef1_massive[-1]}; f1 rouge1: {rouge1f1_massive[-1]}")

luhntf = Luhn(mode="sentences", score_method="tf-idf", ind_coeff=3)

result = luhntf.get_summary_mass(text, 10)

N = len(result)
cum_rouge1 = cum_rouge2 = cumf1 = cum1f1 =  0
summaries_luhntf = []
for i, res in enumerate(result):
    example = gold[i]
    t_r = make_text(firts_n(res, 300))
    summaries_luhntf.append(t_r)

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 2)
    cum_rouge2 += pres
    cumf1 += f1

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 1)
    cum_rouge1 += pres
    cum1f1 += f1

with open('summaries_luhntf.pkl', 'wb') as f:
    pickle.dump(list(zip(summaries_luhntf, gold, text)), f)

rouge1_massive.append(cum_rouge1 / N)
rouge2_massive.append(cum_rouge2 / N)
rougef1_massive.append(cumf1 / N)
rouge1f1_massive.append(cum1f1 / N)
labels.append("Lhun(full) tfidf")
print(f"Lhun (full sentences) tfidf rouge1: {rouge1_massive[-1]}; rouge2: {rouge2_massive[-1]}; f1 rouge2: {rougef1_massive[-1]}; f1 rouge1: {rouge1f1_massive[-1]}")


luhntf_parts = Luhn(mode="parts", score_method="tf-idf", ind_coeff=3)

result = luhntf_parts.get_summary_mass(text, 10)

N = len(result)
cum_rouge1 = cum_rouge2 = cumf1 = cum1f1 =  0
summaries_luhntf_parts = []
for i, res in enumerate(result):
    example = gold[i]
    t_r = make_text(firts_n(res, 300))
    summaries_luhntf_parts.append(t_r)

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 2)
    cum_rouge2 += pres
    cumf1 += f1

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 1)
    cum_rouge1 += pres
    cum1f1 += f1

with open('summaries_luhntf_parts.pkl', 'wb') as f:
    pickle.dump(list(zip(summaries_luhntf_parts, gold, text)), f)

rouge1_massive.append(cum_rouge1 / N)
rouge2_massive.append(cum_rouge2 / N)
rougef1_massive.append(cumf1 / N)
rouge1f1_massive.append(cum1f1 / N)
labels.append("Lhun(parts) tfidf")
print(f"Lhun (parts) tfidf rouge1: {rouge1_massive[-1]}; rouge2: {rouge2_massive[-1]}; f1 rouge2: {rougef1_massive[-1]}; f1 rouge1: {rouge1f1_massive[-1]}")




lexr = TextRank(mode="tf-idf", setted_idf=True)

result = lexr.get_summary_mass(text, 10)

N = len(result)
cum_rouge1 = cum_rouge2 = cumf1 = cum1f1 =  0
summaries_lexr = []
for i, res in enumerate(result):
    example = gold[i]
    t_r = make_text(firts_n(res, 300))
    summaries_lexr.append(t_r)

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 2)
    cum_rouge2 += pres
    cumf1 += f1

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 1)
    cum_rouge1 += pres
    cum1f1 += f1

rouge1_massive.append(cum_rouge1 / N)
rouge2_massive.append(cum_rouge2 / N)
rougef1_massive.append(cumf1 / N)
rouge1f1_massive.append(cum1f1 / N)
labels.append("LexRank")
print(f"LexRank rouge1: {rouge1_massive[-1]}; rouge2: {rouge2_massive[-1]}; f1 rouge2: {rougef1_massive[-1]}; f1 rouge1: {rouge1f1_massive[-1]}")

with open('summaries_lexr.pkl', 'wb') as f:
    pickle.dump(list(zip(summaries_lexr, gold, text)), f)




textr = TextRank(mode="count", setted_idf=True)

result = textr.get_summary_mass(text, 10)
N = len(result)
cum_rouge1 = cum_rouge2 = cumf1 = cum1f1 =  0
summaries_textr = []
for i, res in enumerate(result):
    example = gold[i]
    t_r = make_text(firts_n(res, 300))
    summaries_textr.append(t_r)

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 2)
    cum_rouge2 += pres
    cumf1 += f1

    f1, pres, rec = rougef1(t_r, example, segmenter, morph_tagger, 1)
    cum_rouge1 += pres
    cum1f1 += f1

rouge1_massive.append(cum_rouge1 / N)
rouge2_massive.append(cum_rouge2 / N)
rougef1_massive.append(cumf1 / N)
rouge1f1_massive.append(cum1f1 / N)
labels.append("TextRank")
print(f"TextRank rouge1: {rouge1_massive[-1]}; rouge2: {rouge2_massive[-1]}; f1 rouge2: {rougef1_massive[-1]}; f1 rouge1: {rouge1f1_massive[-1]}")

with open('summaries_textr.pkl', 'wb') as f:
    pickle.dump(list(zip(summaries_textr, gold, text)), f)


fig, ax = plt.subplots()
bars = ax.bar(labels, rouge1_massive)
ax.bar_label(bars, fmt='%.2f', padding=3)
plt.title("ROUGE-1")
plt.ylim(0, 0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rouge1.png")
plt.show()

fig, ax = plt.subplots()
bars = ax.bar(labels, rouge2_massive)
ax.bar_label(bars, fmt='%.2f', padding=3)
plt.title("ROUGE-2")
plt.xticks(rotation=45)
plt.ylim(0, 0.4)
plt.tight_layout()
plt.savefig("rouge2.png")
plt.show()

fig, ax = plt.subplots()
bars = ax.bar(labels, rougef1_massive)
ax.bar_label(bars, fmt='%.2f', padding=3)
plt.title("ROUGE-2 F1")
plt.xticks(rotation=45)
plt.ylim(0, 0.4)
plt.tight_layout()
plt.savefig("rouge2_f1.png")
plt.show()

fig, ax = plt.subplots()
bars = ax.bar(labels, rouge1f1_massive)
ax.bar_label(bars, fmt='%.2f', padding=3)
plt.title("ROUGE-1 F1")
plt.xticks(rotation=45)
plt.ylim(0, 0.4)
plt.tight_layout()
plt.savefig("rouge1_f1.png")
plt.show()