import random
import flair


def augment(sentences, token_list, method, augment_p):
    if method == 'word_replace':
        return random_word_replace(sentences, token_list, augment_p)

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
