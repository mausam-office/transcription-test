import editdistance
import random
import string
from scipy.io import wavfile
from difflib import SequenceMatcher
from configs import (
    WORD_DIGITS_MAPPING,
    KODES,
    DIGITS,
    N_MIDDLE_WORDS,
    MAX_AUDIO_DURATION,
    NEPALI_DIGITS,
    NEP_ENG_DIGITS,
    ENG_NEP_DIGITS,
)


def replaces(text):
    text = text.replace("�", "")

    for letter in string.ascii_letters:
        text = text.replace(letter, "")

    return text


def subword_to_words(text):
    text_splitted = text.split()  # type: ignore
    _text_splitted = []
    for word in text_splitted:
        if word not in KODES + DIGITS:
            _text_splitted.extend(split_to_subwords(word, KODES + DIGITS))
        else:
            _text_splitted.append(word)
    return _text_splitted


def similarity_score(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()


def choose_from_multiple_selected(candidate, matches):
    prefix_matches = []
    suffix_matches = []
    intermediate_matches = []
    for selection in matches:
        if selection.startswith(candidate):
            prefix_matches.append(selection)
        elif selection.endswith(candidate):
            suffix_matches.append(selection)
        else:
            intermediate_matches.append(selection)

    if prefix_matches:
        return random.choice(prefix_matches)
    elif suffix_matches:
        return random.choice(suffix_matches)
    else:
        return random.choice(intermediate_matches)


def find_best_match(candidate, word_dict):
    # best_match = None
    # best_score = 0

    # for edit_distance in sorted(word_dict.keys()):
    #     for word in word_dict[edit_distance]:
    #         score = similarity_score(candidate, word)
    #         if score > best_score:
    #             best_score = score
    #             best_match = word

    score_matches = {}
    for edit_distance in sorted(word_dict.keys()):
        for word in word_dict[edit_distance]:
            score = similarity_score(candidate, word)
            if score_matches.get(score) is None:
                score_matches[score] = [word]
            else:
                score_matches[score].append(word)
    if not list(score_matches.keys()):
        return candidate

    highest_score = max(list(score_matches.keys()))

    best_match_2 = (
        choose_from_multiple_selected(candidate, score_matches[highest_score])
        if len(score_matches[highest_score]) > 1
        else score_matches[highest_score][0]
    )
    # print(f"{candidate = }: {best_match = }: {best_match_2 = }")

    return best_match_2


def closest_word_dist_priority(candidate, vocabulary, thresh_dist=4):

    if candidate in vocabulary:
        return candidate

    dist_closest_words = {i: [] for i in range(thresh_dist + 1)}

    for word in vocabulary:
        distance = editdistance.eval(candidate, word)  # Levenshtein distance

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

    if word_idx_1 in DIGITS[:10]:
        single_digit = True

    # if len(text_splitted) < 5:
    temp = []
    for word in text_splitted:
        if len(word) > 4 and word.endswith("हजार"):
            temp.extend([word.replace("हजार", ""), "हजार"])
        elif (
            len(word) > 2
            and word.endswith("सय")
            and (closest_word_dist_priority(word, DIGITS) != "उनान्सय")
        ):
            temp.extend([word.replace("सय", ""), "सय"])
        else:
            temp.append(word)
    text_splitted = temp

    second_word = (
        closest_word_dist_priority(text_splitted[1], DIGITS[:10])
        if len(text_splitted) >= 2
        else None
    )
    fourth_word = (
        closest_word_dist_priority(text_splitted[3], KODES)
        if len(text_splitted) >= 4
        else None
    )
    fifth_word = (
        closest_word_dist_priority(text_splitted[4], DIGITS[:-2])
        if len(text_splitted) >= 5
        else None
    )

    if (
        single_digit
        and closest_word_dist_priority(text_splitted[3], KODES + DIGITS) not in KODES
    ):
        offset = 1
    elif single_digit and (
        second_word in DIGITS[:10] or fourth_word in KODES or fifth_word in DIGITS[:-2]
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

    prior_digit = text_splitted[:offset]
    middle_words = text_splitted[offset : offset + N_MIDDLE_WORDS]
    posterior_digits = text_splitted[offset + N_MIDDLE_WORDS :]

    text = " ".join(prior_digit + middle_words + posterior_digits)

    prior_digit = [
        closest_word_dist_priority(
            word,
            DIGITS,
            # (
            #     DIGITS[10:]
            #     if (len(prior_digit) == 1 and not single_digit)
            #     else DIGITS[:10]
            # ),
        )
        for word in prior_digit
    ]
    middle_words = [closest_word_dist_priority(word, KODES) for word in middle_words]
    posterior_digits = [
        closest_word_dist_priority(word, DIGITS) for word in posterior_digits
    ]

    text = " ".join(prior_digit + middle_words + posterior_digits)
    return text


def calc_audio_duration(filepath):
    sample_rate, data = wavfile.read(filepath)
    duration = len(data) / sample_rate
    return duration


def has_valid_duration(filepath):
    duration = calc_audio_duration(filepath)
    return True if duration < MAX_AUDIO_DURATION else False


def get_posterior_digits_first_index(kode_chunks):
    middle_word_found = False
    posterior_digits_first_index = None
    for idx, word in enumerate(kode_chunks):
        if isinstance(word, str):
            middle_word_found = True

        if middle_word_found and isinstance(word, int):
            posterior_digits_first_index = idx
            break
    return posterior_digits_first_index


def get_prior_digits_last_index(kode_chunks):
    prior_digits_last_index = None
    for idx, word in enumerate(kode_chunks):
        if isinstance(word, str):
            prior_digits_last_index = idx
            break
    return prior_digits_last_index


def eng_to_nep_digits(kodes_chunks):
    kode_modified_new = []
    for ele in kodes_chunks:
        np_ele = ""
        if isinstance(ele, int):
            ele_str = str(ele)
            if len(ele_str) >= 1:
                for char in ele_str:
                    char = ENG_NEP_DIGITS[int(char)]
                    np_ele += char
                kode_modified_new.append(np_ele)
        else:
            kode_modified_new.append(ele)
    return kode_modified_new


def format_prior_digits(kode_modified: list):
    # find index of digits after middle words
    prior_digits_last_index = get_prior_digits_last_index(kode_modified)

    kode_modified = [str(word) for word in kode_modified]
    val = "".join(kode_modified[:prior_digits_last_index])

    kode_modified = [val] + kode_modified[prior_digits_last_index:]
    kode_modified = [int(word) if word.isnumeric() else word for word in kode_modified]
    return kode_modified


def format_posterior_digits(kode_modified: list, includes_multipler: bool):
    posterior_digits_first_index = get_posterior_digits_first_index(kode_modified)
    if not includes_multipler:
        kode_modified = [str(word) for word in kode_modified]
        val = "".join(kode_modified[posterior_digits_first_index:])
        kode_modified = kode_modified[:posterior_digits_first_index] + [val]
        kode_modified = [
            int(word) if word.isnumeric() else word for word in kode_modified
        ]
    else:
        prior_digits_last_index = (
            idx if (idx := get_prior_digits_last_index(kode_modified)) else 0
        )
        val = 0
        for idx in range(prior_digits_last_index, len(kode_modified)):
            if kode_modified[idx] == 1000:
                if isinstance(kode_modified[idx - 1], int):
                    val += kode_modified[idx - 1] * kode_modified[idx]
                else:
                    val += kode_modified[idx]
            elif kode_modified[idx] == 100:
                if (
                    isinstance(kode_modified[idx - 1], int)
                    and kode_modified[idx - 1] != 1000
                ):
                    val += kode_modified[idx - 1] * kode_modified[idx]
                else:
                    val += kode_modified[idx]
            elif (
                kode_modified[idx - 1] == 1000 or kode_modified[idx - 1] == 100
            ) and kode_modified[idx] == kode_modified[-1]:
                val += kode_modified[idx]

        kode_modified = kode_modified[:posterior_digits_first_index] + [val]

    kode_modified = eng_to_nep_digits(kode_modified)

    return kode_modified


def pad_prior_digits(kode_chunks):
    if len(kode_chunks[0]) != 2:
        n_pad = 2 - len(kode_chunks[0])
        if n_pad < 0:
            print("2 digit at max")
        elif n_pad == 0:
            print("No padding required")
        else:
            for i in range(n_pad):
                kode_chunks[0] = NEPALI_DIGITS[0] + kode_chunks[0]
    return kode_chunks


def pad_posterior_digits(kode_chunks):
    if len(kode_chunks[-1]) != 4:
        n_pad = 4 - len(kode_chunks[-1])
        if n_pad < 0:
            print("4 digit at max")
        elif n_pad == 0:
            print("No padding required")
        else:
            for i in range(n_pad):
                kode_chunks[-1] = NEPALI_DIGITS[0] + kode_chunks[-1]
    return kode_chunks


def replace_words_with_digits(kode_splitted):
    includes_multipler = "सय" in kode_splitted or "हजार" in kode_splitted
    if includes_multipler:
        kode_modified = []

        for idx in range(len(kode_splitted)):
            current_word = kode_splitted[idx]

            if current_word in DIGITS:
                kode_modified.append(NEP_ENG_DIGITS[WORD_DIGITS_MAPPING[current_word]])
            else:
                kode_modified.append(current_word)
    else:
        kode_modified = [
            NEP_ENG_DIGITS[WORD_DIGITS_MAPPING[word]] if word in DIGITS else word
            for word in kode_splitted
        ]
    kode_modified = format_prior_digits(kode_modified)
    kode_modified = format_posterior_digits(kode_modified, includes_multipler)

    return kode_modified


def kataho_code_with_digits(kode: str):
    kode_splitted = kode.strip().split()

    kode_modified = replace_words_with_digits(kode_splitted)

    kode_modified = pad_prior_digits(kode_modified)
    kode_modified = pad_posterior_digits(kode_modified)

    return " ".join(kode_modified)
