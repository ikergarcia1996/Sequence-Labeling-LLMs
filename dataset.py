import itertools
import math
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, List, Literal, Optional, Union

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from label_names import label2name
from tag_encoding import rewrite_labels


def read_tsv(filepath, verbosity: str = True) -> (List[List[str]], List[List[str]]):
    """
    READ tsv file in conll format
    Args:
        filepath: Path to the file
    Returns: List of words, List of labels
    """

    dataset_words: List[List[str]] = []
    dataset_labels: List[List[str]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    dataset_words.append(words)
                    dataset_labels.append(labels)
                    words = []
                    labels = []
            else:
                try:
                    word, label = line.split()
                except ValueError:
                    try:
                        word, label, _ = line.split()
                    except ValueError:
                        if verbosity:
                            print(f"Cannot split line: {line}")
                        continue
                if word:
                    words.append(word)
                    labels.append(label)
        if words:
            dataset_words.append(words)
            dataset_labels.append(labels)

    if verbosity:
        print(f"Read {len(dataset_words)} sentences from {filepath}")

    dataset_labels = [
        rewrite_labels(labels, encoding="iob2") for labels in dataset_labels
    ]

    return dataset_words, dataset_labels


def get_task_tags(filepath, verbosity: str = True):
    dataset_words, dataset_labels = read_tsv(filepath, verbosity=verbosity)
    task_labels = []
    for sentence_labels in dataset_labels:
        for label in sentence_labels:
            if label != "O":
                task_labels.append(label[2:])
    task_labels = list(set(task_labels))
    task_labels = [label2name(label) for label in task_labels]
    start_tags = [f"<{label}>" for label in task_labels]
    end_tags = [f"</{label}>" for label in task_labels]

    # Show the tags
    start_tags.sort()
    end_tags.sort()

    if verbosity:
        print(f"Start tags: {start_tags}")
        print(f"End tags: {end_tags}")

    return start_tags, end_tags


def get_task_labels(filepath, verbosity: str = True):
    dataset_words, dataset_labels = read_tsv(filepath, verbosity=verbosity)
    task_labels = []
    for sentence_labels in dataset_labels:
        for label in sentence_labels:
            if label != "O":
                task_labels.append(label[2:])
    task_labels = list(set(task_labels))
    task_labels.sort()
    if verbosity:
        print(f"Task labels: {task_labels}")
    return task_labels


def compute_words_ids_old(tokenizer: PreTrainedTokenizerBase, sentence: str):
    if tokenizer.is_fast:
        tokenized_sentence = tokenizer([sentence], add_special_tokens=False)
        words_ids = tokenized_sentence.word_ids()

    else:
        words = sentence.split()
        words_ids = []
        for word_no, word in enumerate(words):
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            words_ids.extend([word_no] * len(word_ids))
    return words_ids


def compute_words_ids(tokenizer: PreTrainedTokenizerBase, sentence: str):
    # if tokenizer.is_fast and "llama" not in tokenizer.name_or_path.lower():
    #    # LLaMA does not compute word_ids, see: https://github.com/huggingface/transformers/issues/25082
    #    tokenized_sentence = tokenizer([sentence], add_special_tokens=False)
    #    words_ids = tokenized_sentence.word_ids()
    # else:
    tokenized_sentence = tokenizer(sentence, add_special_tokens=False).input_ids
    current_char = 0
    current_word = 0
    words_ids = []
    for token_id in tokenized_sentence:
        words_ids.append(current_word)
        token = tokenizer.decode(token_id).strip()
        current_char += len(token)
        if current_char + 1 < len(sentence) and sentence[current_char] == " ":
            current_word += 1
            current_char += 1

    return words_ids


def auto_detect_if_we_need_to_add_spaces_around_tags(
    tokenizer: PreTrainedTokenizerBase, verbosity: str = True
) -> str:
    """
    Auto-detect how we need to format the target sentence depending on the tokenizer. Modes:
        - "together": The president<Person>Obama</Person>went to<Location>New York</Location>.
        - "after": The president<Person> Obama</Person> went to<Location>New York </Location>.
        - "both": The president <Person> Obama </Person>went to <Location> New York </Location>.
    Args:
        tokenizer: Model tokenizer
    Returns: tokenizer mode: "together", "after" or "both"
    """

    # Special case for T5 tokenizer
    # This tokenzier is weird, I tested it and not
    # adding spaces around tags gives better results
    # So we will always return "together" for T5 tokenizer
    if "t5" in tokenizer.__class__.__name__.lower():
        if verbosity:
            print(
                "We have auto-detected that the tokenizer is a T5 tokenizer.\n"
                "We will tokenize the target sentence as follows: <Person>Obama</Person>went to<Location>New York</Location>.\n"
                "If the contrained F1 score is lower than expected, or the unconstrained F1 score is higher than the constrained F1 score, "
                "it is probably related to the tokenization of the target sentence. Open an issue on the GitHub repository or"
                "manually edit the `auto_detect_if_we_need_to_add_spaces_around_tags` function in the `dataset.py` file."
            )
        return "together"

    def compare_tokenizations(labeled: List[int], unlabeled: List[int]):
        """
        Test if all the tokens in 'unlabeled' are present in 'labeled'
        Also test if no whitespace token is present in labeled
        Also test if the labels are in the sentence

        Args:
            labeled: List of token ids of the labeled sentence
            unlabeled: List of token ids of the unlabeled sentence

        Returns: True if all tokens are present in labeled and no whitespace token is present in labeled
        """
        # print(labeled)
        # print(unlabeled)
        for token in unlabeled:
            if token not in labeled:
                return False
            else:
                # Remove
                labeled.remove(token)

        # Check if labels are in the sentence
        label_s = tokenizer.encode("<Person>", add_special_tokens=False)
        label_e = tokenizer.encode("</Person>", add_special_tokens=False)
        for token in label_s:
            if token not in labeled:
                return False
            else:
                labeled.remove(token)
        for token in label_e:
            if token not in labeled:
                return False
            else:
                labeled.remove(token)

        # Check if there are no whitespace tokens
        for token in labeled:
            if tokenizer.decode(token).strip() == "":
                return False

        return True

    unlabeled = "President Obama president"
    unlabeled = tokenizer.encode(unlabeled, add_special_tokens=False)

    # Test together
    labeled = "President<Person>Obama</Person>president"
    labeled = tokenizer.encode(labeled, add_special_tokens=False)

    if compare_tokenizations(labeled, unlabeled):
        if verbosity:
            print(
                "We have auto-detected that the tokenizer for the model requires no whitespace around tags.\n"
                "We will tokenize the target sentence as follows: <Person>Obama</Person>went to<Location>New York</Location>.\n"
                "If the contrained F1 score is lower than expected, or the unconstrained F1 score is higher than the constrained F1 score, "
                "it is probably related to the tokenization of the target sentence. Open an issue on the GitHub repository or "
                "manually edit the `auto_detect_if_we_need_to_add_spaces_around_tags` function in the `dataset.py` file."
            )
        return "together"

    # Test after
    labeled = "President<Person> Obama</Person> president"
    labeled = tokenizer.encode(labeled, add_special_tokens=False)
    if compare_tokenizations(labeled, unlabeled):
        if verbosity:
            print(
                "We have auto-detected that the tokenizer for the model requires a whitespace after tags.\n"
                "We will tokenize the target sentence as follows: <Person> Obama</Person> went to<Location> New York</Location>.\n"
                "If the contrained F1 score is lower than expected, or the unconstrained F1 score is higher than the constrained F1 score, "
                "it is probably related to the tokenization of the target sentence. Open an issue on the GitHub repository or"
                "manually edit the `auto_detect_if_we_need_to_add_spaces_around_tags` function in the `dataset.py` file."
            )
        return "after"

    # Test both
    labeled = "President <Person> Obama </Person> president"
    labeled = tokenizer.encode(labeled, add_special_tokens=False)
    if compare_tokenizations(labeled, unlabeled):
        if verbosity:
            print(
                "We have auto-detected that the tokenizer for the model requires a whitespace before and after tags.\n"
                "We will tokenize the target sentence as follows: <Person> Obama </Person> went to <Location> New York </Location>.\n"
                "If the contrained F1 score is lower than expected, or the unconstrained F1 score is higher than the constrained F1 score, "
                "it is probably related to the tokenization of the target sentence. Open an issue on the GitHub repository or "
                "manually edit the `auto_detect_if_we_need_to_add_spaces_around_tags` function in the `dataset.py` file."
            )
        return "both"
    if verbosity:
        print(
            "WARNING!!! We could not auto-detect the correct tokenization mode for the target sentence. "
            "We will use the default mode and add whitespaces around tags.\n"
            "Here is an example <Person> Obama </Person> went to <Location> New York </Location>.\n"
            "But this may not be the correct tokenization for the model. If the model does not perform well, "
            "you may need to manually edit the `auto_detect_if_we_need_to_add_spaces_around_tags` function in the `dataset.py` file.\n"
            "You can also open an issue on the GitHub repository."
        )
    return "both"


def format_label(
    label: str, is_start: bool, format: Literal["together", "after", "both"]
):
    """
    Format label for seq2seq models
    Args:
        label: "PER"
        is_start: Whether the label is a start label
        format: "together", "after" or "both"
    Returns: Formatted label
    """
    if is_start:
        if format == "together":
            return f"<{label}>"
        elif format == "after":
            return f"<{label}> "
        elif format == "both":
            return f" <{label}> "
    else:
        if format == "together":
            return f"</{label}>"
        elif format == "after":
            return f"</{label}> "
        elif format == "both":
            return f" </{label}> "


def format_target_sentence(
    words: List[str], labels: List[str], format: Literal["together", "after", "both"]
) -> (str, str):
    """
       Format target sentence for seq2seq models
    Args:
        words: ["Obama","went","to","New","York", "."]
        labels: ["B-PER","O","O","B-LOC","I-LOC","O"]
        format: "together", "after" or "both"
    Returns: Tuple with the following elements:
        - Original sentence
        - Formatted target sentence
            If format is "together":
                <PER>Obama</PER>went to<LOC>New York</LOC>.
            If format is "after":
                <PER> Obama</PER> went to <LOC> New York</LOC> .
            If format is "both":
                <PER> Obama </PER> went to <LOC> New York </LOC> .

    """

    target = []
    inside_entity: bool = False
    prev_label: str = ""
    prev_is_word = False
    for word, label in zip(words, labels):
        if label == "O":
            if inside_entity:
                target.append(format_label(prev_label, is_start=False, format=format))
                prev_is_word = False
                inside_entity = False
            if prev_is_word:
                target.append(" ")
            target.append(word)
            prev_is_word = True
        elif label.startswith("B-"):
            if inside_entity:
                target.append(format_label(prev_label, is_start=False, format=format))
                prev_is_word = False
                inside_entity = False
            prev_label = label2name(label[2:])
            target.append(format_label(prev_label, is_start=True, format=format))
            target.append(word)
            prev_is_word = True
            inside_entity = True
        elif label.startswith("I-"):
            if prev_is_word:
                target.append(" ")
            target.append(word)
            prev_is_word = True
        else:
            raise ValueError(
                f"Unknown label: {label}\nwords: {words}\nlabels: {labels}"
            )

    if inside_entity:
        target.append(format_label(prev_label, is_start=False, format=format))

    # Special case for after format
    # If the sentence starts by a label, we need to remove the whitespace after the label
    # else the id of the first token will change
    # ('AL', 3702) ('-', 20) ('AIN', 208497)
    # If we add whitespace ('<Location>', 255031) (' AL', 17405) ('-', 20) ('AIN', 208497)
    # If we do not add whitespace ('<Location>', 255031) ('AL', 3702) ('-', 20) ('AIN', 208497)

    if format == "after":
        if labels[0] != "O":
            target[0] = target[0].strip(" ")

    return " ".join(words).strip(), "".join(target).strip()


def prepare_sl(
    tokenizer: PreTrainedTokenizerBase,
    add_spaces_around_tags: Literal["together", "after", "both"],
    words: List[str],
    labels: List[str],
    max_source_len: int,
    max_target_len: int,
    is_encoder_decoder: bool,
    train: bool = True,
    input_prompt: Optional[str] = None,
) -> BatchEncoding:
    """
    Prepare data for seq2seq model
    Args:
        tokenizer: Model tokenizer
        add_spaces_around_tags: Format for the target sentence
        words: List of words in the sentence we want to label
        labels: List of gold labels for each word
        max_source_len: Max length of the source sentence
        max_target_len: Max length of the target sentence
        is_encoder_decoder: Whether the model is encoder-decoder or decoder only
        train: Whether we are preparing data for training or inference
        input_prompt: Prompt to append at the beginning of the input
    Returns: Dictionary with the following keys:
        - input_ids: Input ids for the encoder
        - attention_mask: Attention mask for the encoder
        - labels: Target ids to predict
        - original_sentence_ids: Original sentence ids for each token
        - labeled_sentence_ids: Labeled sentence ids for each token
        - words_ids: Words ids for each token
    """

    source_sentence, target_sentence = format_target_sentence(
        words, labels, add_spaces_around_tags
    )

    encoder_inputs = (
        f"{input_prompt.strip(' ')} {source_sentence.strip(' ')}"
        if input_prompt
        else source_sentence
    )
    encoder_inputs_original = encoder_inputs

    if is_encoder_decoder:
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    if tokenizer.chat_template is not None:
        # print("Chat template found in the tokenizer. We will apply it to the input.")
        encoder_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": encoder_inputs}],
            tokenize=False,
            add_generation_prompt=True,
        )

    if is_encoder_decoder:
        labels = target_sentence

    else:
        if train:
            if tokenizer.chat_template is not None:
                labels = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": encoder_inputs_original},
                        {"role": "assistant", "content": target_sentence.strip(" ")},
                    ],
                    tokenize=False,
                )
                decoder_prompt = encoder_inputs
                encoder_inputs = labels
            else:
                labels = f"{encoder_inputs.strip(' ')} -> {target_sentence.strip(' ')}"
                decoder_prompt = f"{encoder_inputs.strip(' ')} ->"
                encoder_inputs = (
                    f"{encoder_inputs.strip(' ')} -> {target_sentence.strip(' ')}"
                )

        else:
            if tokenizer.chat_template is not None:
                labels = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": encoder_inputs_original},
                        {"role": "assistant", "content": target_sentence.strip(" ")},
                    ],
                    tokenize=False,
                )
                decoder_prompt = encoder_inputs
                encoder_inputs = encoder_inputs
            else:
                labels = f"{encoder_inputs.strip(' ')} -> {target_sentence.strip(' ')}"
                decoder_prompt = f"{encoder_inputs.strip(' ')} ->"
                encoder_inputs = f"{encoder_inputs.strip(' ')} ->"

    model_inputs = tokenizer(
        text=encoder_inputs,
        max_length=max_source_len,
        padding=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=tokenizer.chat_template is None,
    )

    if is_encoder_decoder:
        y_tokenized = tokenizer(
            text_target=labels,
            max_length=max_target_len,
            padding=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=tokenizer.chat_template is None,
        )

        if train:
            model_inputs["loss_weight_mask"] = np.ones(
                len(y_tokenized["input_ids"]), dtype=np.float32
            )

    else:
        y_tokenized = tokenizer(
            text=labels,
            max_length=max_target_len,
            padding=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=tokenizer.chat_template is None,
        )

        if train:
            # Make sure the `eos_token_id` is added at the end
            # This bug is reported at https://github.com/huggingface/transformers/issues/22794
            if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
                model_inputs["input_ids"].append(tokenizer.eos_token_id)
                model_inputs["attention_mask"].append(1)
                y_tokenized["input_ids"].append(tokenizer.eos_token_id)
        else:
            # Remove the last token if it is an eos token
            if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
                model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:-1]

        # Get len of the prompt
        prompt = tokenizer(
            text=decoder_prompt,
            max_length=max_source_len,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=tokenizer.chat_template is None,
        )["input_ids"]

        # Remove the last token if it is an eos token
        if prompt[-1] == tokenizer.eos_token_id:
            prompt = prompt[:-1]

        if len(prompt) >= len(y_tokenized["input_ids"]):
            raise ValueError(
                f"Prompt is longer than the input, something went wrong. Prompt: {prompt}, input:"
                f" {y_tokenized['input_ids']}.\n"
                f"Prompt: {tokenizer.decode(prompt)}\n"
                f"Input: {tokenizer.decode(y_tokenized['input_ids'])}. \n"
                f"The most probable cause is that the input is too long and was truncated,"
                f" increase the max_source_len and try again."
            )

        # Create the weight mask
        loss_weight_mask = np.ones(len(y_tokenized["input_ids"]), dtype=np.float32)
        for i in range(len(prompt)):
            loss_weight_mask[i] = 0.0

        model_inputs["loss_weight_mask"] = loss_weight_mask

    model_inputs["labels"] = y_tokenized["input_ids"].copy()

    model_inputs["original_sentence_ids"] = tokenizer(
        text=source_sentence,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    if is_encoder_decoder:
        model_inputs["labeled_sentence_ids"] = tokenizer(
            text_target=target_sentence,
            padding=False,
            truncation=False,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
    else:
        model_inputs["labeled_sentence_ids"] = tokenizer(
            text=target_sentence,
            padding=False,
            truncation=False,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

    model_inputs["words_ids"] = compute_words_ids(tokenizer, source_sentence)

    return model_inputs


def batch(iterable, n=1) -> iter:
    l: int = len(iterable)
    p: int = math.ceil(l / n)
    for ndx in range(0, l, p):
        yield iterable[ndx : min(ndx + p, l)]


def batch_tokenization(
    tokenizer: PreTrainedTokenizerBase,
    add_spaces_around_tags: bool,
    max_source_len: int,
    max_target_len: int,
    is_encoder_decoder: bool,
    train: bool,
    input_prompt: Optional[str],
    batch_words: List[List[str]],
    batch_labels: List[List[str]],
    process_no: int,
    verbosity: str = True,
):
    dataset = []
    for words, labels in zip(
        tqdm(
            batch_words,
            desc=f"Data tokenization {process_no}",
            leave=True,
            disable=not verbosity,
        ),
        batch_labels,
    ):
        assert len(words) == len(labels)
        dataset.append(
            prepare_sl(
                tokenizer,
                add_spaces_around_tags,
                words,
                labels,
                max_source_len,
                max_target_len,
                is_encoder_decoder,
                train,
                input_prompt,
            )
        )

    return dataset


class SequenceLabellingDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        file_path: str,
        max_source_len: int,
        max_target_len: int,
        is_encoder_decoder: bool,
        train: bool = True,
        input_prompt: Optional[str] = None,
        num_workers: int = 8,
        add_labels_as_context: bool = False,
        verbosity: str = True,
    ):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.is_encoder_decoder = is_encoder_decoder
        self.train = train
        self.task_labels = get_task_labels(filepath=file_path, verbosity=verbosity)
        self.start_tags, self.end_tags = get_task_tags(
            filepath=file_path, verbosity=verbosity
        )
        self.start_tags_original = self.start_tags.copy()
        # Add labels with space prefix
        self.start_tags += [f" {tag}" for tag in self.start_tags]
        self.end_tags += [f" {tag}" for tag in self.end_tags]
        # self.task_labels += self.task_labels

        add_spaces_around_tags = auto_detect_if_we_need_to_add_spaces_around_tags(
            tokenizer, verbosity=verbosity
        )

        self.start_labels_ids = [
            tokenizer.encode(tag, add_special_tokens=False) for tag in self.start_tags
        ]
        self.end_labels_ids = [
            tokenizer.encode(tag, add_special_tokens=False) for tag in self.end_tags
        ]
        if verbosity:
            print(f"Start labels ids: {self.start_labels_ids}")
            print(f"End labels ids: {self.end_labels_ids}")

        if add_labels_as_context:
            if input_prompt:
                input_prompt = (
                    f"{input_prompt} {' '.join(self.start_tags_original)} ".strip(" ")
                )
            else:
                input_prompt = f"{' '.join(self.start_tags_original)} ".strip(" ")

        dataset_words, dataset_labels = read_tsv(file_path, verbosity=verbosity)

        if verbosity:
            print(
                f"Tokenizing {len(dataset_words)} sentences with {num_workers} workers"
            )

        with Pool(num_workers) as p:
            dataset = p.starmap(
                batch_tokenization,
                zip(
                    itertools.repeat(tokenizer),
                    itertools.repeat(add_spaces_around_tags),
                    itertools.repeat(max_source_len),
                    itertools.repeat(max_target_len),
                    itertools.repeat(is_encoder_decoder),
                    itertools.repeat(train),
                    itertools.repeat(input_prompt),
                    batch(dataset_words, num_workers),
                    batch(dataset_labels, num_workers),
                    range(num_workers),
                    itertools.repeat(verbosity),
                ),
            )

        self.dataset = list(itertools.chain.from_iterable(dataset))

        if verbosity:
            print(f"Dataset size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].copy()


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        inputs_ids = (
            [feature["input_ids"] for feature in features]
            if "input_ids" in features[0].keys()
            else None
        )
        max_input_len = max(len(l) for l in inputs_ids)

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        orig_labels = (
            [feature["labels"].copy() for feature in features].copy()
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        words_ids = (
            [feature["words_ids"] for feature in features]
            if "words_ids" in features[0].keys()
            else None
        )
        if words_ids is not None:
            max_words_ids_length = max(len(l) for l in words_ids)
            for feature in features:
                remainder = [-1] * (max_words_ids_length - len(feature["words_ids"]))
                feature["words_ids"] = feature["words_ids"] + remainder

        original_sentence_ids = (
            [feature["original_sentence_ids"] for feature in features]
            if "original_sentence_ids" in features[0].keys()
            else None
        )
        if original_sentence_ids is not None:
            max_original_sentence_ids_length = max(
                len(l) for l in original_sentence_ids
            )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_original_sentence_ids_length
                    - len(feature["original_sentence_ids"])
                )
                feature["original_sentence_ids"] = (
                    feature["original_sentence_ids"] + remainder
                )

        labeled_sentence_ids = (
            [feature["labeled_sentence_ids"] for feature in features]
            if "labeled_sentence_ids" in features[0].keys()
            else None
        )
        if labeled_sentence_ids is not None:
            max_labeled_sentence_ids_length = max(len(l) for l in labeled_sentence_ids)
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_labeled_sentence_ids_length
                    - len(feature["labeled_sentence_ids"])
                )
                feature["labeled_sentence_ids"] = (
                    feature["labeled_sentence_ids"] + remainder
                )

        loss_weight_mask = (
            [feature["loss_weight_mask"] for feature in features]
            if "loss_weight_mask" in features[0].keys()
            else None
        )

        if loss_weight_mask is not None:
            max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
            if self.pad_to_multiple_of is not None:
                max_loss_weight_mask_length = (
                    (max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
                    max_loss_weight_mask_length - len(feature["loss_weight_mask"])
                )
                if isinstance(feature["loss_weight_mask"], list):
                    feature["loss_weight_mask"] = (
                        feature["loss_weight_mask"] + remainder
                        if padding_side == "right"
                        else remainder + feature["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_weight_mask"] = np.concatenate(
                        [feature["loss_weight_mask"], remainder]
                    ).astype(np.float32)
                else:
                    feature["loss_weight_mask"] = np.concatenate(
                        [remainder, feature["loss_weight_mask"]]
                    ).astype(np.float32)

        # print(self.tokenizer.padding_side)
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        """
        if features["input_ids"].size() != features["labels"].size():
            raise ValueError(
                f"Input and label sizes do not match\n"
                f"Input size: {features['input_ids'].size()}\n"
                f"Label size: {features['labels'].size()}\n"
                f"max_input_len: {max_input_len}\n"
                f"max_label_length: {max_label_length}\n"
                f""
                f"Input: {features['input_ids']}\n"
                f"Label: {features['labels']}\n"
                f"Input: {self.tokenizer.batch_decode(inputs_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)}\n"
                f"Label: {self.tokenizer.batch_decode(orig_labels,skip_special_tokens=False,clean_up_tokenization_spaces=False)}\n"
            )
        """
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


def get_dataloader(
    tokenizer,
    filenames,
    batch_size,
    max_source_len,
    max_target_len,
    is_encoder_decoder,
    train,
    input_prompt,
    num_workers,
    add_labels_as_context,
    verbosity: bool = True,
):
    if len(filenames) == 1:
        dataset = SequenceLabellingDataset(
            tokenizer=tokenizer,
            file_path=filenames[0],
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            is_encoder_decoder=is_encoder_decoder,
            train=train,
            input_prompt=input_prompt,
            num_workers=num_workers,
            add_labels_as_context=add_labels_as_context,
            verbosity=verbosity,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,  # = 8 May be faster on some hardware
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=data_collator,
            pin_memory=True,
        )

        return dataloader

    else:
        datasets = []
        for filename in filenames:
            datasets.append(
                SequenceLabellingDataset(
                    tokenizer=tokenizer,
                    file_path=filename,
                    max_source_len=max_source_len,
                    max_target_len=max_target_len,
                    is_encoder_decoder=is_encoder_decoder,
                    train=train,
                    input_prompt=input_prompt,
                    num_workers=num_workers,
                    add_labels_as_context=add_labels_as_context,
                    verbosity=verbosity,
                )
            )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=None,  # = 8 May be faster on some hardware
        )

        concatenated_dataset = ConcatDataset(datasets)

        dataloader = DataLoader(
            concatenated_dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=data_collator,
            pin_memory=True,
        )

        return dataloader
