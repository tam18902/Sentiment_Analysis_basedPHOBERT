from ultis import *
import underthesea

def word_to_tokenize(text, sw):
    for id, line in enumerate(text):
        line = underthesea.word_tokenize(line)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        text[id] = underthesea.word_tokenize(line, format="text")

sw = load_stopwords()
split = ["test","train","validation"]
for s in split:
    v_text, v_label, v_topic = load_data(f"raw_data/data_{s}.csv")
    word_to_tokenize(v_text, sw)
    with open(f"preprocess_{s}.csv","w") as f:
        for id, text in enumerate(v_text):
            if len(text) == 0:
                continue
            f.write(f"{text},{v_label[id]},{v_label[id]}\n")