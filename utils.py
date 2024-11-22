import editdistance

from difflib import SequenceMatcher
from configs import KODES, DIGITS, N_MIDDLE_WORDS

def similarity_score(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()


def find_best_match(candidate, word_dict):
    best_match = None
    best_score = 0
    
    for edit_distance in sorted(word_dict.keys()):
        for word in word_dict[edit_distance]:
            score = similarity_score(candidate, word)
            if score > best_score:
                best_score = score
                best_match = word
        
        # If a match is found at the smallest edit distance, break early
        if best_match and edit_distance == 1:
            break
    
    return best_match

    
def closest_word_dist_priority(candidate, vocabulary, thresh_dist=4):
    
    if candidate in vocabulary:
        return candidate

    dist_closest_words = {i:[] for i in range(thresh_dist+1)}

    for word in vocabulary:
        distance = editdistance.eval(candidate, word)   # Levenshtein distance
        
        if 0 <= distance <= thresh_dist:
            dist_closest_words[distance].append(word)
    
    best_match = find_best_match(candidate, dist_closest_words)

    return best_match if best_match else candidate


def split_to_subwords(word, vocabulary):
    n = len(word)
    
    for i in range(1, n):
        prefix = word[:i]
        suffix = word[i:]
        if prefix in vocabulary and suffix in vocabulary:
            return [prefix, suffix]
    return [word]


def process(text_splitted, single_digit=False):
    # Apply edit-distance
    word_idx_1 = closest_word_dist_priority(text_splitted[0], DIGITS)
    
    if (word_idx_1 in DIGITS[:10]):
        single_digit = True
        # words.append(word_idx_1)
        
    if len(text_splitted) < 5:
        temp = []
        for word in text_splitted:
            if len(word)>4 and word.endswith('हजार'):
                temp.extend([word.replace('हजार', ''), 'हजार'])
            elif len(word)>2 and word.endswith('सय') and (
                closest_word_dist_priority(word, DIGITS)!='उनान्सय'
                ):
                temp.extend([word.replace('सय', ''), 'सय'])
            else:
                temp.append(word)
        text_splitted = temp
    
    second_word = closest_word_dist_priority(text_splitted[1], DIGITS[:10]) if len(text_splitted)>=2 else None 
    fourth_word = closest_word_dist_priority(text_splitted[3], KODES) if len(text_splitted)>=4 else None
    fifth_word = closest_word_dist_priority(text_splitted[4], DIGITS[:-2]) if len(text_splitted)>=5 else None
    
    if (single_digit and 
        (second_word in DIGITS[:10] or fourth_word in KODES or fifth_word in DIGITS[:-2])
    ):
        # ! prior_digit_indices = [0, 1]
        # ! middle_words_indices = [2, 3]
        # ! posterior_digit_indices = [4 and more]
        offset = 2
    else:
        # ! prior_digit_indices = [0]
        # ! middle_words_indices = [1, 2]
        # ! posterior_digit_indices = [3 and more]
        offset = 1
        
    prior_digit = text_splitted[ : offset]
    middle_words = text_splitted[offset : offset+N_MIDDLE_WORDS]
    posterior_digits = text_splitted[offset+N_MIDDLE_WORDS : ]
        
    text = ' '.join(prior_digit + middle_words + posterior_digits)
    
    prior_digit = [
        closest_word_dist_priority(
            word, DIGITS[10:] if len(prior_digit)==1 else DIGITS[:10]
        ) 
        for word in prior_digit
    ]
    middle_words = [closest_word_dist_priority(word, KODES) for word in middle_words]
    posterior_digits = [closest_word_dist_priority(word, DIGITS) for word in posterior_digits]
    
    text = ' '.join(prior_digit + middle_words + posterior_digits)
    return text
