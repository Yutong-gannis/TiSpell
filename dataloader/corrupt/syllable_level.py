import random
from typing import List
from .utils import split_syllable


def syllable_random_delete(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()
    syllable_list_mask = syllable_list.copy()
    if len(syllable_list) <= 1:
        return syllable_list, syllable_list_mask

    delete_index = random.randint(0, len(syllable_list) - 1)
    del syllable_list_copy[delete_index]
    syllable_list_mask[delete_index] = "[MASK]"
    return syllable_list_copy, syllable_list_mask

def syllable_random_exchange(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()
    syllable_list_mask = syllable_list.copy()
    if len(syllable_list) < 2:
            return syllable_list, syllable_list_mask
    index1, index2 = random.sample(range(len(syllable_list)), 2)
    syllable_list_copy[index1], syllable_list_copy[index2] = syllable_list_copy[index2], syllable_list_copy[index1]
    syllable_list_mask[index1] = "[MASK]"
    syllable_list_mask[index2] = "[MASK]"
    return syllable_list_copy, syllable_list_mask

def syllable_random_merge(syllable_list: List[str]) -> List[str]:
    syllable_list_copy = syllable_list.copy()
    if len(syllable_list) < 2:
        return syllable_list

    index = random.randint(0, len(syllable_list) - 2)
    syllable_list_copy[index] = syllable_list_copy[index] + syllable_list_copy[index + 1]
    del syllable_list_copy[index + 1]
    return syllable_list_copy


if __name__ == '__main__':
    text_example = "ཞི་ཅིན་ཕིང་གིས་ཏི་ས་ནཱ་ཡ་ཁེས་སི་རི་ལན་ཁའི་ཙུང་ཐུང་གི་འགན་བཞེས་པར་རྟེན་འབྲེལ་གློག་འཕྲིན་བཏང་གནང་བ།"
    syllable_list = split_syllable(text_example)
    print(syllable_random_delete(syllable_list))
    print(syllable_random_exchange(syllable_list))
    print(syllable_random_merge(syllable_list))
