from typing import List
import re
from label_names import name2label, label2name
import os
from seqeval.metrics import classification_report, f1_score
from tqdm.auto import tqdm


def split_sentence(
    tag_regex,
    sentence: str,
    recursion_limit: int = 100,
) -> List[str]:
    sentence = sentence.strip().split()

    if recursion_limit == 0:
        return sentence

    new_sentence: List[str] = []

    for word in sentence:
        search_result = tag_regex.search(word)
        if search_result:
            span = search_result.span()

            l = word[: span[0]].strip()
            r = word[span[1] :].strip()
            t = word[span[0] : span[1]].strip()
            if l:
                new_sentence.extend(split_sentence(tag_regex, l, recursion_limit - 1))
            new_sentence.append(t)
            if r:
                new_sentence.extend(split_sentence(tag_regex, r, recursion_limit - 1))

        else:
            new_sentence.append(word)

    return new_sentence


def get_label_type(label: str) -> (str, bool):
    label = label.strip()
    is_start = not label.startswith("</")
    if is_start:
        label_type = name2label(label[1:-1])
    else:
        label_type = name2label(label[2:-1])

    return label_type, is_start


def get_iob(html_sentence: str, possible_labels: List[str]) -> (List[str], List[str]):
    """
    Input
    <Person> Obama </Person> went to <Location> New York </Location> .
    Output
    ["B-PER","O","O","B-LOC","I-LOC","O"]
    """
    inside_tag: bool = False
    current_label_type: str = ""
    tag_regex = re.compile(
        f"</?({'|'.join([label2name(p) for p in possible_labels])})>"
    )

    html_words = split_sentence(
        tag_regex, html_sentence, recursion_limit=max(len(html_sentence) * 8, 999)
    )

    labels = []
    words = []
    first = True

    for word in html_words:
        result = tag_regex.match(word)
        if result:
            label_type, is_start = get_label_type(word)
            if is_start:
                inside_tag = True
                current_label_type = label_type
                first = True
            else:
                inside_tag = False
        else:
            if inside_tag:
                if first:
                    labels.append(f"B-{current_label_type}")
                    first = False
                else:
                    labels.append(f"I-{current_label_type}")

            else:
                labels.append("O")
            words.append(word)

    assert len(words) == len(labels), (
        f"Len of words and labels are not equal: {len(words)} != {len(labels)}\n"
        f"Words: {words}\n"
        f"Labels: {labels}"
    )

    return words, labels


def evaluate_most_probable(
    predictions: List[List[str]],
    gold: List[str],
    output_name: str,
    task_labels: List[str],
):
    output_name = os.path.abspath(output_name)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    predicted_labels = []
    gold_labels = []

    with open(f"{output_name}.tsv", "w", encoding="utf8") as output_file:
        for predicted_sentence, gold_sentence in zip(
            tqdm(predictions, desc="Evaluate Top1"), gold
        ):
            sentence_predicted_words, sentence_predicted_labels = get_iob(
                html_sentence=predicted_sentence[0],
                possible_labels=task_labels,
            )

            sentence_gold_words, sentence_gold_labels = get_iob(
                html_sentence=gold_sentence,
                possible_labels=task_labels,
            )

            if len(sentence_predicted_words) > len(sentence_gold_words):
                # print(
                #    f"Warning. Predicted sentence is longer than gold sentence. "
                #    f"We will truncate the predicted sentence\n"
                #    f"Predicted: {sentence_predicted_words}\n"
                #    f"Gold: {sentence_gold_words}\n"
                # )
                sentence_predicted_words = sentence_predicted_words[
                    : len(sentence_gold_words)
                ]
                sentence_predicted_labels = sentence_predicted_labels[
                    : len(sentence_gold_words)
                ]

            elif len(sentence_predicted_words) < len(sentence_gold_words):
                # print(
                #    f"Warning. Predicted sentence is shorter than gold sentence."
                #    f"We will extend the predicted labels with O's\n"
                #    f"Predicted: {sentence_predicted_words}\n"
                #    f"Gold: {sentence_gold_words}\n"
                # )

                sentence_predicted_labels = sentence_predicted_labels + (
                    ["O"] * (len(sentence_gold_words) - len(sentence_predicted_words))
                )

            for word, tag in zip(sentence_predicted_words, sentence_predicted_labels):
                print(f"{word} {tag}", file=output_file)
            print(file=output_file)

            predicted_labels.append(sentence_predicted_labels)
            gold_labels.append(sentence_gold_labels)

    with open(f"{output_name}.txt", "w", encoding="utf8") as output_file:
        try:
            cr = classification_report(
                y_true=gold_labels, y_pred=predicted_labels, digits=4, zero_division="1"
            )
        except ValueError as e:
            cr = str(e)
        print(
            cr,
            file=output_file,
        )
        try:
            micro_f1 = f1_score(
                y_true=gold_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division="1",
            )
        except ValueError as e:
            print(f"Error calculating micro f1: {e}")
            micro_f1 = 0

        try:
            macro_f1 = f1_score(
                y_true=gold_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division="1",
            )
        except ValueError as e:
            print(f"Error calculating macro f1: {e}")
            macro_f1 = 0
        print(f"Micro F1: {micro_f1}", file=output_file)
        print(f"Macro F1: {macro_f1}", file=output_file)

    return micro_f1


def evaluate_best_prediction(
    predictions: List[List[str]],
    gold: List[str],
    output_name: str,
    task_labels: List[str],
):
    output_name = os.path.abspath(output_name)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    predicted_labels = []
    gold_labels = []

    with open(f"{output_name}.tsv", "w", encoding="utf8") as output_file:
        for predicted_sentence, gold_sentence in zip(
            tqdm(predictions, desc="Evaluate Upperbound"), gold
        ):
            best_pred = 0
            best_pred_f1 = 0

            sentence_gold_words, sentence_gold_labels = get_iob(
                html_sentence=gold_sentence,
                possible_labels=task_labels,
            )

            for i in range(len(predicted_sentence)):
                sentence_predicted_words, sentence_predicted_labels = get_iob(
                    html_sentence=predicted_sentence[i],
                    possible_labels=task_labels,
                )

                if len(sentence_predicted_words) > len(sentence_gold_words):
                    sentence_predicted_labels = sentence_predicted_labels[
                        : len(sentence_gold_words)
                    ]

                elif len(sentence_predicted_words) < len(sentence_gold_words):
                    sentence_predicted_labels = sentence_predicted_labels + (
                        ["O"]
                        * (len(sentence_gold_words) - len(sentence_predicted_words))
                    )

                try:
                    micro_f1 = f1_score(
                        y_true=[sentence_gold_labels],
                        y_pred=[sentence_predicted_labels],
                        average="micro",
                        zero_division="1",
                    )
                except ValueError as e:
                    print(f"Error calculating micro f1: {e}")
                    micro_f1 = 0

                if micro_f1 > best_pred_f1:
                    best_pred = i
                    best_pred_f1 = micro_f1

            sentence_predicted_words, sentence_predicted_labels = get_iob(
                html_sentence=predicted_sentence[best_pred],
                possible_labels=task_labels,
            )

            if len(sentence_predicted_words) > len(sentence_gold_words):
                sentence_predicted_words = sentence_predicted_words[
                    : len(sentence_gold_words)
                ]
                sentence_predicted_labels = sentence_predicted_labels[
                    : len(sentence_gold_words)
                ]
                # print(
                #    f"Warning. Predicted sentence is longer than gold sentence. "
                #    f"We will truncate the predicted sentence\n"
                #    f"Predicted: {sentence_predicted_words}\n"
                #    f"Gold: {sentence_gold_words}\n"
                # )
            elif len(sentence_predicted_words) < len(sentence_gold_words):
                sentence_predicted_labels = sentence_predicted_labels + (
                    ["O"] * (len(sentence_gold_words) - len(sentence_predicted_words))
                )
                # print(
                #    f"Warning. Predicted sentence is shorter than gold sentence."
                #    f"We will extend the predicted labels with O's\n"
                #    f"Predicted: {sentence_predicted_words}\n"
                #    f"Gold: {sentence_gold_words}\n"
                # )

            for word, tag in zip(sentence_predicted_words, sentence_predicted_labels):
                print(f"{word} {tag}", file=output_file)
            print(file=output_file)

            predicted_labels.append(sentence_predicted_labels)
            gold_labels.append(sentence_gold_labels)

    with open(f"{output_name}.txt", "w", encoding="utf8") as output_file:
        try:
            cr = classification_report(
                y_true=gold_labels, y_pred=predicted_labels, digits=4, zero_division="1"
            )
        except ValueError as e:
            cr = str(e)
        print(
            cr,
            file=output_file,
        )
        try:
            micro_f1 = f1_score(
                y_true=gold_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division="1",
            )
        except ValueError as e:
            print(f"Error calculating micro f1: {e}")
            micro_f1 = 0

        try:
            macro_f1 = f1_score(
                y_true=gold_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division="1",
            )
        except ValueError as e:
            print(f"Error calculating macro f1: {e}")
            macro_f1 = 0
        print(f"Micro F1: {micro_f1}", file=output_file)
        print(f"Macro F1: {macro_f1}", file=output_file)

    return micro_f1
