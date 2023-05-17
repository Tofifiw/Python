import re


def count_words(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # zły pomysł; proszę to zrobić z Wikipedią
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        word_counts = {}
        for word in words:  # collections.Counter
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [sorted_word_counts[0]]
        for i in range(1, len(sorted_word_counts)):
            if sorted_word_counts[i][1] == top_words[-1][1]:
                top_words.append(sorted_word_counts[i])
            else:
                if len(top_words) >= n:
                    break
                top_words.append(sorted_word_counts[i])
        return top_words


file_path = 'potop.txt'
n = 10
top_words = count_words(file_path, n)

print(f'Najczęściej występujące słowa ({n}), razem z remisami:')
for word, count in top_words:
    print(f'{word}: {count}')
