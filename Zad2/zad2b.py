import re

def count_ngrams(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
        text = re.sub(r'[^\w\s]','',text)
        words = text.split()
        # wyklucza słowa kończące się na znaki specjalne
        words = [re.sub(r'[\.\,\;\:\?\!]+$', '', word) for word in words]
        ngrams = {}
        for i in range(len(words)-n+1):
            ngram = ' '.join(words[i:i+n])
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
        sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
        return sorted_ngrams


file_path = 'potop.txt'
# n można zmienić na dowolną cyfrę (dla bi- lub trigramów również)
n = 2
top_ngrams = count_ngrams(file_path, n)

print(f'The top {n}-grams are:')
for ngram, count in top_ngrams:
    print(f'{ngram}: {count}')