import random
from typing import List

from .letters import TIBETAN_LETTER, ROOT_LETTER, ROOT_LETTER_SHORT, HOMOMORPHIC_LETTER, short_to_tall, tall_to_short
from .utils import split_syllable


def char_random_delete(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()
    if not syllable_list:
        return syllable_list

    syllable_index = random.randint(0, len(syllable_list) - 1)
    selected_syllable = syllable_list[syllable_index]

    if not selected_syllable:
        return syllable_list

    char_index = random.randint(0, len(selected_syllable) - 1)
    deleted_syllable = selected_syllable[:char_index] + selected_syllable[char_index + 1:]
    syllable_list_copy[syllable_index] = deleted_syllable
    return syllable_list_copy

def char_random_insert(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()
    if not syllable_list:
        return syllable_list
    
    syllable_index = random.randint(0, len(syllable_list) - 1)
    selected_string = syllable_list[syllable_index]
    
    insert_index = random.randint(0, len(selected_string))
    
    random_char = random.choice(TIBETAN_LETTER)
    modified_string = selected_string[:insert_index] + random_char + selected_string[insert_index:]
    syllable_list_copy[syllable_index] = modified_string
    return syllable_list_copy

def char_tall_short_replace(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()
    if not syllable_list:
        return syllable_list

    syllable_index = random.randint(0, len(syllable_list) - 1)
    selected_syllable = syllable_list[syllable_index]

    if not selected_syllable:
        return syllable_list

    char_index = random.randint(0, len(selected_syllable) - 1)
    selected_char = selected_syllable[char_index]

    if selected_char in ROOT_LETTER:
        new_char = tall_to_short(selected_char)
    elif selected_char in ROOT_LETTER_SHORT:
        new_char = short_to_tall(selected_char)
    else:
        new_char = selected_char

    modified_string = selected_syllable[:char_index] + new_char + selected_syllable[char_index + 1:]
    syllable_list_copy[syllable_index] = modified_string
    return syllable_list_copy


def char_homomorphic_replace(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()

    syllables_with_replacements = [
        s for s in syllable_list_copy if any(char in HOMOMORPHIC_LETTER for char in s)
    ]

    if not syllables_with_replacements:
        return syllable_list_copy

    selected_syllable = random.choice(syllables_with_replacements)
    syllable_index = syllable_list_copy.index(selected_syllable)

    replaceable_indices = [i for i, char in enumerate(selected_syllable) if char in HOMOMORPHIC_LETTER]
    char_index = random.choice(replaceable_indices)
    selected_char = selected_syllable[char_index]

    new_char = random.choice(HOMOMORPHIC_LETTER[selected_char])

    modified_string = selected_syllable[:char_index] + new_char + selected_syllable[char_index + 1:]
    syllable_list_copy[syllable_index] = modified_string

    return syllable_list_copy

def char_inner_syllable_exchange(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()

    long_syllables = [s for s in syllable_list_copy if len(s) > 2]

    if not long_syllables:
        return syllable_list_copy

    selected_syllable = random.choice(long_syllables)
    syllable_index = syllable_list_copy.index(selected_syllable)

    idx1, idx2 = random.sample(range(len(selected_syllable)), 2)

    char_list = list(selected_syllable)
    char_list[idx1], char_list[idx2] = char_list[idx2], char_list[idx1]

    modified_string = ''.join(char_list)

    syllable_list_copy[syllable_index] = modified_string
    return syllable_list_copy

def char_near_syllable_exchange(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()

    if len(syllable_list_copy) < 2:
        return syllable_list_copy

    i = random.randint(0, len(syllable_list_copy) - 2)
    syllable1 = syllable_list_copy[i]
    syllable2 = syllable_list_copy[i + 1]

    if not syllable1 or not syllable2:
        return syllable_list_copy

    char_index1 = random.randint(0, len(syllable1) - 1)
    selected_char1 = syllable1[char_index1]

    char_index2 = random.randint(0, len(syllable2) - 1)
    selected_char2 = syllable2[char_index2]

    modified_syllable1 = syllable1[:char_index1] + selected_char2 + syllable1[char_index1 + 1:]
    modified_syllable2 = syllable2[:char_index2] + selected_char1 + syllable2[char_index2 + 1:]

    syllable_list_copy[i] = modified_syllable1
    syllable_list_copy[i + 1] = modified_syllable2

    return syllable_list_copy

if __name__ == '__main__':
    text_example = "ཞི་ཅིན་ཕིང་གིས་ཏི་ས་ནཱ་ཡ་ཁེས་སི་རི་ལན་ཁའི་ཙུང་ཐུང་གི་འགན་བཞེས་པར་རྟེན་འབྲེལ་གློག་འཕྲིན་བཏང་གནང་བ།"
    syllable_list = split_syllable(text_example)
    print(char_random_delete(syllable_list))
    print(char_random_insert(syllable_list))
    print(char_tall_short_replace(syllable_list))
    print(char_homomorphic_replace(syllable_list))
    print(char_inner_syllable_exchange(syllable_list))
    print(char_near_syllable_exchange(syllable_list))
