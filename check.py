import pickle

with open('summaries_textr.pkl', 'rb') as f:
    data = pickle.load(f)

for summaries, gold, text in data[:4]:

    print(f"text \n{text}\n")
    print(f"summary \n{summaries}\n")
    print(f"gold \n{gold}\n")
    print("-" * 100)