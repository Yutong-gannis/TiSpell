# Prefix letters
PERFIX_LETTER = ["ག", "ད", "བ", "མ", "འ"]

# Root letters used as the main base characters in Tibetan script
ROOT_LETTER = ["ཀ", "ཁ", "ག", "ང",
               "ཅ", "ཆ", "ཇ", "ཉ",
               "ཏ", "ཐ", "ད", "ན",
               "པ", "ཕ", "བ", "མ",
               "ཙ", "ཚ", "ཛ", "ཝ",
               "ཞ", "ཟ", "འ", "ཡ",
               "ར", "ལ", "ཤ", "ས", "ཧ", "ཨ"]

# Short forms of root letters used when stacked with superscript letters
ROOT_LETTER_SHORT= ['ྐ', 'ྑ', 'ྒ', 'ྔ',
                     'ྕ', 'ྖ', 'ྗ', 'ྙ',
                     'ྟ', 'ྠ', 'ྡ', 'ྣ',
                     'ྤ', 'ྥ', 'ྦ', 'ྨ',
                     'ྩ', 'ྪ', 'ྫ', 'ྮ',
                     'ྯ', 'ྰ', 'ླ', 'ྴ',
                     'ྶ', 'ྷ', 'ྸ']

# Mapping of short root letters to their standard tall forms
SHORT_TO_TALL = {'ྐ': "ཀ", 'ྑ': "ཁ", 'ྒ': "ག", 'ྔ': "ང",
                  'ྕ': "ཅ", 'ྖ': "ཆ", 'ྗ': "ཇ", 'ྙ': "ཉ",
                  'ྟ': "ཏ", 'ྠ': "ཐ", 'ྡ': "ད", 'ྣ': "ན",
                  'ྤ': "པ", 'ྥ': "ཕ", 'ྦ': "བ", 'ྨ': "མ",
                  'ྩ': "ཙ", 'ྪ': "ཚ", 'ྫ': "ཛ", 'ྮ': "ཞ",
                  'ྯ': "ཟ", 'ྰ': "འ", 'ླ': "ལ", 'ྴ': "ཤ",
                  'ྶ': "ས", 'ྷ': "ཧ", 'ྸ': "ཨ"}

# Superscript letters
SUPERSCRIPT_LETTER = ["ར", "ལ", "ས"]

# Subscript letters
SUBSCRIPT_LETTER = ["ྱ", "ྲ", "ླ", "ྭ", "ཱ"]

# Farther subscript letters
FARTHER_SUBSCRIPT_LETTER = ["ྭ", "ཱ"]

# Suffix letters
SUFFIX_LETTER = ["ག", "ང", "ད", "ན", "བ", "མ", "འ", "ར", "ལ", "ས"]

# Farther suffix letters
FARTHER_SHUFFIX_LETTER = ["ད", "ས"]

# Vowel
VOWEL = ["ི", "ུ", "ེ", 'ོ']

STOP = ["།"]

# vernacular
VERNACULAR = ['༺', '༻']

# Reduplication
REDUPLICATION = ["གྷ", "ཌྷ", "ཀྵ", "ཛྷ", "བྷ", "དྷ"]

# Punctuation marks commonly used in Tibetan writing
PUNCTUATION = [" ", ".", ",", ":", ";", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "*", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "`", "~", "<", ">", 
                "·", "，", "。", "、", "；", "：", "？", "！", "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "「", "」", "『", "』", "﹃", "﹄", "〔", "〕", "——", "……", "—", "－", "【", "】"]

# Numbers used in Tibetan numerals
NUMBER = ["༠", "༡", "༢", "༣", "༤", "༥", "༦", "༧", "༨", "༩", "༪", "༫", "༬", "༭", "༮", "༯", "༰", "༱", "༲", "༳"]

# All the letters
TIBETAN_LETTER = ROOT_LETTER + ROOT_LETTER_SHORT + SUBSCRIPT_LETTER + FARTHER_SUBSCRIPT_LETTER + VOWEL + NUMBER

# Arabic numerals
ARABIC_NUMERALS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Other symbols
OTHERS = ['༼', '༽', '༜', '༙', '#', '$']

ROOT_LETTER = ["ཀ", "ཁ", "ག", "ང",
               "ཅ", "ཆ", "ཇ", "ཉ",
               "ཏ", "ཐ", "ད", "ན",
               "པ", "ཕ", "བ", "མ",
               "ཙ", "ཚ", "ཛ", "ཝ",
               "ཞ", "ཟ", "འ", "ཡ",
               "ར", "ལ", "ཤ", "ས", "ཧ", "ཨ"]

# Homomorphic letters
HOMOMORPHIC_LETTER = {"ཀ":["ག", "ཁ"],
                      "ཁ":["ཀ", "ག"],
                      "ག":["ཀ", "ཁ"],
                      "ང":["ད", "ཏ", "ར"],
                      "ཅ":["ཙ"],
                      "ཆ":["ཚ"],
                      "ཇ":["ཛ", "ཟ"],
                      "ཏ":["ང", "ད"],
                      "ཐ":["ཟ"],
                      "ད":["ང", "ཏ"],
                      "པ":["བ", "ཕ"],
                      "ཕ":["པ", "བ"],
                      "བ":["པ", "ཕ"],
                      "ཙ":["ཅ"],
                      "ཚ":["ཆ"],
                      "ཛ":["ཇ", "ཟ"],
                      "ཟ":["ཐ", "ཇ", "ཛ"],
                      "ར":["ང"],
                      'ྐ':['ྑ', 'ྒ'],
                      'ྑ':['ྐ', 'ྒ'],
                      'ྔ':['ྡ'],
                      'ྕ':['ྩ'],
                      'ྖ':['ྪ'],
                      'ྗ':['ྠ', 'ྫ', 'ྯ'],
                      'ྠ':['ྗ', 'ྯ'],
                      'ྤ':['ྥ', 'ྦ'],
                      'ྥ':['ྤ', 'ྦ'],
                      'ྦ':['ྤ', 'ྥ'],
                      'ྩ':['ྕ'],
                      'ྪ':['ྖ'],
                      'ྫ':['ྗ'],
                      "ི":["ེ"],
                      "ེ":["ི"]}

# [属格助词，饰集词，离合词，终结词，具格助词，la类助词]
PACKED_WORD = ["འི", "འང", "འམ", "འོ", "ས", "ར"]

# Relationships defining which root letters can follow specific prefix letters
PREFIX_ROOT_RELATION = {'ག': ['ཅ', 'ཏ', 'ཙ', 'ཉ', 'ད', 'ན', 'ཞ', 'ཟ', 'ཡ', 'ཤ', 'ས'],
                        'ད': ['ཀ', 'པ', 'ག', 'བ', 'ང', 'མ'],
                        'བ': ['ཀ', 'ཅ', 'ཏ', 'ཙ', 'ག', 'ང', 'ཇ', 'ཉ', 'ད', 'ན', 'ཛ', 'ཞ', 'ཟ', 'ཪ', 'ཤ', 'ས'],
                        'མ': ['ཁ', 'ཆ', 'ཐ', 'ཚ', 'ག', 'ཇ', 'ད', 'ཛ', 'ང', 'ཉ', 'ན'],
                        'འ': ['ག', 'ཇ', 'ད', 'བ', 'ཛ', 'ཁ', 'ཆ', 'ཐ', 'ཕ', 'ཚ']}
'''
SUPERSCRIPT_ROOT_RELATION = {'ར': ['ྐ', 'ྒ', 'ྔ', 'ྗ', 'ྟ', 'ྡ', 'ྣ', 'ྦ', 'ྨ', 'ྩ', 'ྫ', 'ྙ'],
                             'ལ': ['ྐ', 'ྒ', 'ྔ', 'ྕ', 'ྗ', 'ྟ', 'ྡ', 'ྤ', 'ྦ', 'ྷ'],
                             'ས': ['ྐ', 'ྒ', 'ྔ', 'ྙ', 'ྟ', 'ྡ', 'ྣ', 'ྤ', 'ྦ', 'ྨ', 'ྩ']}
'''

# Relationships for valid combinations of superscript and root letters
SUPERSCRIPT_ROOT_RELATION = {'ར': ["ཀ", "ག", "ང", "ཇ", "ཏ", "ད", "ན", "བ", "མ", "ཙ", "ཛ", "ཉ"],
                             'ལ': ["ཀ", "ག", "ང", "ཅ", "ཇ", "ཏ", "ད", "པ", "བ", "ཧ"],
                             'ས': ["ཀ", "ག", "ང", "ཉ", "ཏ", "ད", "ན", "པ", "བ", "མ", "ཙ"]}

# Relationships for valid combinations of subscript and root letters
SUBSCRIPT_ROOT_RELATION = {"ྱ": ['ཀ', 'ཁ', 'ག', 'པ', 'ཕ', 'བ', 'མ'],
                           "ྲ": ['ཀ', 'ཁ', 'ག', 'ཏ', 'ཐ', 'ད', 'པ', 'ཕ', 'བ', 'མ', 'ས', 'ཧ'],
                           "ླ": ['ཀ', 'ག', 'བ', 'ཟ', 'ཪ', 'ས'],
                           "ྭ": ['ཀ', 'ཁ', 'ག', 'ཉ', 'ད', 'ཙ', 'ཚ', 'ཞ', 'ཟ', 'ར', 'ལ', 'ཤ', 'ཧ'],
                           "ཱ": ROOT_LETTER}

# Relationships defining which suffix letters can follow specific farther suffix letters
SUFFIX_RELATION = {'ད': ['ན', 'ར', 'ལ'],
                   'ས': ['ག', 'ང', 'བ', 'མ']}


def short_to_tall(short: str) -> str:
    tall = SHORT_TO_TALL[short]
    return tall

def tall_to_short(tall: str) -> str:
    for key, value in SHORT_TO_TALL.items():
        if tall == value:
            return key
    return tall