import editdistance


def calc_cer(target_text, predicted_text) -> float:
    words_target = target_text.split()
    if len(words_target) == 0:
        return 1.
    return editdistance.distance(words_target, predicted_text.split()) / len(words_target)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.
    return editdistance.distance(target_text, predicted_text) / len(target_text)
