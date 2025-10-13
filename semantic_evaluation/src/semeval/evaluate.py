"""
Semantic Evaluation script.

Computes Precision, Recall and F1 Measures based on SemEval dataset.
The metrics can be computed as complete and affixes segmentation. The
affixes segmentation also considers a full word as a possible correct
segmentation as it allows its meaning to emerge from a single token.
"""

import csv

from transformers import AutoTokenizer


def generate_predicted_positions(tokenizer, word):
    '''Convert the tokenizer segments into a sequence of boundary positions.'''
    unk_id = tokenizer.unk_token_id
    tokenized = tokenizer(
        word,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    # unknown words
    if unk_id in tokenized['input_ids']:
        print(f'>>> {word}')

    offsets = tokenized['offset_mapping']
    splits = [o[0] for o in offsets[1:]]
    return splits


def generate_target_positions(tokens: list[str]):
    '''Convert the SemEval segmentation in a sequence of boundary positions.'''
    if len(tokens) == 1:
        return []

    position = len(tokens[0])
    positions = [position]
    for token in tokens[1:-1]:
        position += len(token)
        positions.append(position)

    return positions


def compute_confusion_matrix(target, predicted, affix_only=False, is_suffix=True):
    '''Computes the confusion matrix statistics.'''
    # considers a single word (no predicted split) as a true positive case
    if affix_only and len(predicted) == 0:
        return 1, 0, 0

    t_index = 0
    p_index = 0

    tp = 0
    fp = 0
    fn = 0

    if affix_only:
        if is_suffix:
            # skip predictions that are not suffixes
            while p_index < len(predicted) and target[0] > predicted[p_index]:
                p_index += 1
        else:
            # remove predictions that are not prefixes
            while len(predicted) > 0 and predicted[-1] > target[-1]:
                predicted.pop()

    # while both target a predicted have splits
    while t_index < len(target) and p_index < len(predicted):
        t = target[t_index]
        p = predicted[p_index]

        if t < p:       # false negative
            fn += 1
            t_index += 1
            continue

        if p < t :      # false positive
            fp += 1
            p_index += 1
            continue

        # p == t : true positive
        tp += 1
        t_index += 1
        p_index += 1

    # process remaining false negatives
    while t_index < len(target):
        fn += 1
        t_index += 1

    # process remaining false positives
    while p_index < len(predicted):
        fp += 1
        p_index += 1

    return tp, fp, fn


def evaluate(dataset_path, model_name):
    '''Semantic Evaluation of Tokenizers'''
    print(f"{model_name} ========")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    words = 0
    sw = 0
    tp_f = 0
    fp_f = 0
    fn_f = 0
    tp_a = 0
    fp_a = 0
    fn_a = 0

    # for line in samples:
    with open(dataset_path, 'r', encoding='utf-8') as dataset_file:
        input_reader = csv.reader(dataset_file, delimiter='\t')
        for line in input_reader:
            words += 1
            word = line[0]
            target = line[1]
            is_suffix = True if len(line) == 2 else True if line[2] == 'suffix' else False

            t_tokens = target.split('|')
            t_positions = generate_target_positions(t_tokens)
            p_positions = generate_predicted_positions(tokenizer, word)

            # predicted as a single word
            if len(p_positions) == 0:
                sw += 1

            _tp_f, _fp_f, _fn_f = compute_confusion_matrix(t_positions, p_positions, affix_only=False)
            _tp_a, _fp_a, _fn_a = compute_confusion_matrix(t_positions, p_positions, affix_only=True, is_suffix=is_suffix)

            tp_f += _tp_f
            fp_f += _fp_f
            fn_f += _fn_f
            tp_a += _tp_a
            fp_a += _fp_a
            fn_a += _fn_a

    precision_f = tp_f / (tp_f + fp_f)
    recall_f = tp_f / (tp_f + fn_f)
    f1_f = 2 * precision_f * recall_f  / (precision_f + recall_f)

    precision_a = tp_a / (tp_a + fp_a)
    recall_a = tp_a / (tp_a + fn_a)
    f1_a = 2 * precision_a * recall_a  / (precision_a + recall_a)

    print("\nFULL Metrics")
    print(f"Precision: {100 * precision_f:.2f}%")
    print(f"Recall: {100 * recall_f:.2f}%")
    print(f"F1: {100 * f1_f:.2f}%")

    print("\nAffix Metrics")
    print(f"Precision: {100 * precision_a:.2f}%")
    print(f"Recall: {100 * recall_a:.2f}%")
    print(f"F1: {100 * f1_a:.2f}%")
    print(f"Single Words: {sw}/{words} ({100 * sw / words:.2f}%)")
