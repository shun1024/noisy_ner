import random
import flair


def augment(sentences, token_list, method, augment_p):
    if method == 'word_replace':
        return random_word_replace(sentences, token_list, augment_p)

    if method == 'char_replace':
        return random_char_replace(sentences, token_list, augment_p)

    if method == 'mask_replace':
        return mask_replace(sentences, augment_p)
    return None


def mask_replace(sentences, p):
    result = []
    for i in range(len(sentences)):
        sentence = flair.data.Sentence()
        for token in sentences[i]:
            text = token.text
            if random.random() < p:
                text = '[MASK]'
            nt = flair.data.Token(text)
            sentence.add_token(nt)
        result.append(sentence)
    return result


def random_char_replace(sentences, token_list, p):
    char_list = []
    for token in token_list:
        for char in token:
            char_list.append(char)

    char_list = list(set(char_list))
    char_length = len(char_list)

    result = []
    for i in range(len(sentences)):
        sentence = flair.data.Sentence()
        for token in sentences[i]:
            text = token.text
            text_list = list(text)
            if random.random() < p:
                text_list[random.randint(0, len(text_list)-1)] = char_list[random.randint(0, char_length-1)]
            nt = flair.data.Token("".join(text_list))
            sentence.add_token(nt)
        result.append(sentence)
    return result


def random_word_replace(sentences, token_list, p):
    result = []
    for i in range(len(sentences)):
        sentence = flair.data.Sentence()
        for token in sentences[i]:
            text = token.text
            if random.random() < p:
                text = token_list[random.randint(0, len(token_list) - 1)]
            nt = flair.data.Token(text)
            sentence.add_token(nt)
        result.append(sentence)
    return result
