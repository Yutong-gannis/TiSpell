import re
from typing import List


def split_syllable(text: str) -> List[str]:
    if type(text) != str: 
        raise ValueError("\"text\" is not a string, please input a string.")
    word_list = re.split('་|༌|།|༎|༏|༐|༑', text)
    return word_list

def punctuation_clean(text: str) -> str:
    if type(text) != str: 
        raise ValueError("\"text\" is not a string, please input a string.")
    sentences = re.split('།|༎|༏|༐|༑', text)
    sentences = [item for item in sentences if item != ""]
    return sentences