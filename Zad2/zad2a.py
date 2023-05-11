from collections import Counter


def count_words(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
        word_count = Counter(words)
        most_common = word_count.most_common(n)

        # szuka wszystkie słowa z tą samą częstotliwością występowania co n
        i = n
        while i < len(word_count) and word_count.most_common(i+1)[-1][1] == most_common[-1][1]:
            i += 1
        ties = word_count.most_common(i)

        return ties

file_path = 'potop.txt'
n = 6
top_words = count_words(file_path, n)

print(f'Najczęściej występujące słowa ({n}), razem z remisami:')
for word, count in top_words:
    print(f'{word}: {count}')