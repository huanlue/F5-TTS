import re
import pykakasi
from janome.tokenizer import Tokenizer
kakasi = pykakasi.kakasi()
tokenizer = Tokenizer()
def preprocess_japanese(text):
    """
    使用 Janome 对日语文本进行分词，并转化为平假名。
    """
    # 将文本转换为平假名
    result = kakasi.convert(text)
    hiragana_text = "".join([item['hira'] for item in result])
    # 去除标点符号
    # hiragana_text = re.sub(r"[、。]", "", hiragana_text)

    # 使用 Janome 分词
    tokenized_text = " ".join([token.surface for token in tokenizer.tokenize(hiragana_text)])
    text = "".join(list(hiragana_text))
    return text

if __name__ == "__main__":
    text = '''映像があるので、それを皆さんにお伝えしたいと思います。'''
    preprocessed_text = preprocess_japanese(text)
    print(preprocessed_text)