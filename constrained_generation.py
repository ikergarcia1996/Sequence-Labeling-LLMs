import copy
from typing import Dict, List, Union

import torch
from transformers import BatchEncoding, PreTrainedModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


def compute_words_ids(tokenizer, sentence):
    words = sentence.split()
    words_ids = []
    for word_no, word in enumerate(words):
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        words_ids.extend([word_no] * len(word_ids))
    return words_ids


class TrieNode:
    def __init__(self):
        self.labels_ids: Dict[int, TrieNode] = {}

    def add_label(self, label_ids: List[int], label_no: int):
        if len(label_ids) == 0:
            self.labels_ids = label_no
            return
        if label_ids[0] not in self.labels_ids:
            self.labels_ids[label_ids[0]] = TrieNode()
        self.labels_ids[label_ids[0]].add_label(label_ids[1:], label_no)


class LabelTrie:
    """
    Prefix tree for labels.
    """

    def __init__(self, labels_ids: List[List[int]], labels_names):
        self.root = TrieNode()
        assert len(labels_ids) == len(labels_names)
        self.labels_ids = labels_ids
        self.path = []
        self.label_names = labels_names
        for label_ids, label_name in zip(labels_ids, labels_names):
            self.root.add_label(label_ids, label_name)

        self.pointer = self.root

    def get_next_labels(self):
        if isinstance(self.pointer.labels_ids, int):
            return self.pointer.labels_ids
        return list(self.pointer.labels_ids.keys())

    def move_to_label(self, label_id):
        if label_id not in self.pointer.labels_ids:
            raise ValueError(
                f"Label {label_id} not in trie, unable to move to it. "
                f"Path: {self.path}. Labels: {self.pointer.labels_ids}"
            )
        self.pointer = self.pointer.labels_ids[label_id]
        self.path.append(label_id)

    def reset(self):
        self.pointer = self.root
        self.path = []

    def copy(self):
        trie = LabelTrie(self.labels_ids, self.label_names)
        for label_id in self.path:
            trie.move_to_label(label_id)
        return trie

    def next_is_last(self, label_id):
        if isinstance(self.pointer.labels_ids[label_id].labels_ids, int):
            return self.pointer.labels_ids[label_id].labels_ids
        return -1


class SequenceState:
    """
    Represent the current state of the sequence generation.
    current_token_index: The index of the current token in the input sequence.
    current_label_start: The id of the open label, if there is no open label, the value is -1.
    current_label_token_index: The index of the current token in the label.
        STATES:
        - next_outside: The model can choose between the next token or an opening tag.
        - next_inside: The model can choose between the next token or a closing tag.
        - complete_word_outside: The model can only choose to complete current word (we are not inside a label).
        - complete_word_inside: The model can only choose to complete current word (we are inside a label).
        - complete_label_start: The model can only choose to complete the current start label.
        - complete_label_end: The model can only choose to complete the current end label.
        - end: The model can only choose to generate the pad token
    """

    def __init__(
        self,
        current_token_index: int = 0,
        current_label_id: int = -1,
        current_seq_len: int = 0,
        current_label_token_index: int = 0,
        seqlen: int = 0,
        state: str = "next_outside",
    ):
        self.current_token_index = current_token_index
        self.current_label_id = current_label_id
        self.state = state
        self.generated_seq_len = current_seq_len
        self.seqlen = seqlen
        self.current_label_token_index = current_label_token_index

    def report_state(self):
        print(
            f"=== Sequence State ===\n"
            f"state: {self.state}\n"
            f"current_token_index: {self.current_token_index}\n"
            f"current_label_start: {self.current_label_id}\n"
            f"current_label_token_index: {self.current_label_token_index}\n"
            f"generated_seq_len: {self.generated_seq_len}\n"
            f"seqlen: {self.seqlen}\n"
        )

    def copy(self):
        return SequenceState(
            current_token_index=self.current_token_index,
            current_label_id=self.current_label_id,
            current_seq_len=self.generated_seq_len,
            current_label_token_index=self.current_label_token_index,
            seqlen=self.seqlen,
            state=self.state,
        )

    def new_state(
        self,
        state: str,
        advance_current_token_index: bool = False,
        advance_current_label_token_index: bool = False,
        do_not_advance_seq_len: bool = False,
        current_label_id: int = -1,
    ):
        new_state = self.copy()
        if advance_current_token_index:
            new_state.current_token_index += 1
            new_state.generated_seq_len += 1
        if advance_current_label_token_index:
            new_state.current_label_token_index += 1

        if not do_not_advance_seq_len:
            new_state.seqlen += 1

        new_state.current_label_id = current_label_id
        new_state.state = state

        # Satiny resets
        if (
            new_state.state != "complete_label_start"
            and new_state.state != "complete_label_end"
        ):
            new_state.current_label_token_index = 0

        if (
            new_state.state == "next_outside"
            or new_state.state == "complete_word_outside"
            or new_state.state == "end"
        ):
            new_state.current_label_id = -1

        return new_state


class SequenceLabellingConstraint:
    """
     A constraint that forces the model generates a valid sequence of labels. This class is used to
     generate at each step the valid next tokens and the resulting state of the sequence generation.
    Input: Obama went to Washington.
    Expected Output: <Person> Obama </Person> went to <Place> Washington </Place>.

    At each step, the model can only chose from the next word or an opening tag.
    If a tag is chosen, in the next step, the model can only chose the next word.
    After this word, the model can only chose between the next word or closing the tag.
    """

    def __init__(
        self,
        tokens_ids: Union[List[int], torch.Tensor],
        word_ids: List[int],
        start_labels_ids: List[List[int]],  # Label is following a word
        end_labels_ids: List[List[int]],
        start_labels_names: List[int],
        end_labels_names: List[int],
        pad_token_id: int,
        eos_token_id: int,
    ):
        word_ids = [x for x in word_ids if x is None or x != -1]
        tokens_ids = (
            tokens_ids if isinstance(tokens_ids, list) else tokens_ids.cpu().tolist()
        )
        tokens_ids = [x for x in tokens_ids if x != pad_token_id]

        # Add EOS token if not present
        if tokens_ids[-1] != eos_token_id:
            tokens_ids.append(eos_token_id)
        if word_ids[-1] is not None:
            word_ids.append(None)

        if len(tokens_ids) != len(word_ids):
            raise ValueError(
                f"The length of the sentence_tokens_ids and word_ids should be the same. "
                f"Got {len(tokens_ids)} and {len(word_ids)}\n."
                f"tokens_ids: {tokens_ids}\n"
                f"word_ids: {word_ids}\n"
            )

        if len(start_labels_ids) != len(end_labels_ids):
            raise ValueError(
                f"The length of the start_labels_ids and end_labels_ids should be the same. "
                f"Got {len(start_labels_ids)} and {len(end_labels_ids)}"
            )

        self.tokens_ids = tokens_ids
        self.word_ids = word_ids

        self.start_labels_ids = start_labels_ids
        self.start_labels_names = start_labels_names
        self.start_labels_trie = LabelTrie(
            labels_ids=start_labels_ids,
            labels_names=start_labels_names,
        )

        self.end_labels_ids = end_labels_ids
        self.end_labels_names = end_labels_names
        self.end_labels_dict = {}
        for end_label_ids, end_label_name in zip(end_labels_ids, end_labels_names):
            if end_label_name not in self.end_labels_dict:
                self.end_labels_dict[end_label_name] = []
            self.end_labels_dict[end_label_name].append(end_label_ids)

        for end_label_name in self.end_labels_dict:
            self.end_labels_dict[end_label_name] = self.end_labels_trie = LabelTrie(
                labels_ids=self.end_labels_dict[end_label_name],
                labels_names=[end_label_name]
                * len(self.end_labels_dict[end_label_name]),
            )

        self.pad_token_id = pad_token_id if pad_token_id else eos_token_id
        self.eos_token_id = eos_token_id
        self.current_seq = []
        self.current_state = SequenceState()
        self.seqlen = len(self.tokens_ids)

    def report_state(self):
        print(
            f"=== Constraint State ===\n"
            f"current_seq: {self.current_seq}\n"
            f"seqlen: {self.seqlen}\n"
            f"tokens_ids: {self.tokens_ids}\n"
            f"word_ids: {self.word_ids}\n"
            f"start_labels_ids: {self.start_labels_ids}\n"
            f"end_labels_ids: {self.end_labels_ids}\n"
            f"pad_token_id: {self.pad_token_id}\n"
            f"eos_token_id: {self.eos_token_id}"
        )
        self.current_state.report_state()

    def next_is_subtoken(self) -> bool:
        if self.current_state.current_token_index + 1 > len(self.word_ids) - 1:
            return False
        else:
            return (
                self.word_ids[self.current_state.current_token_index]
                == self.word_ids[self.current_state.current_token_index + 1]
            )

    def advance(self, report=False) -> (torch.Tensor, List[SequenceState]):
        """
        Returns the next possible tokens and the corresponding states.
        Returns: A tuple of the next possible tokens and the corresponding states.
        """
        if report:
            self.report_state()

        if (
            self.current_state.state == "end"
            or self.current_state.current_token_index >= len(self.tokens_ids)
        ):
            "Generation ended, return PAD token"
            return (
                torch.tensor([self.pad_token_id]),
                [
                    self.current_state.new_state(
                        "end",
                        advance_current_token_index=False,
                        current_label_id=-1,
                        do_not_advance_seq_len=True,
                    )
                ],
            )

        elif self.current_state.state == "next_outside":
            """
            The model can choose between the next token or an opening tag.
            """
            # If the next token is a EOS token we don't allow opening a new tag

            if (
                (
                    self.tokens_ids[self.current_state.current_token_index]
                    == self.eos_token_id
                    and self.current_state.current_token_index > 0
                )
                or self.tokens_ids[self.current_state.current_token_index]
                == self.pad_token_id
                or self.word_ids[self.current_state.current_token_index] is None
            ):
                return (
                    torch.tensor(
                        [self.tokens_ids[self.current_state.current_token_index]]
                    ),
                    [
                        self.current_state.new_state(
                            "end",
                            advance_current_token_index=True,
                            current_label_id=-1,
                        )
                    ],
                )
            else:
                # The model can choose between the next token or an opening tag.

                next_label_ids = self.start_labels_trie.get_next_labels()

                return (
                    torch.tensor(
                        [
                            self.tokens_ids[self.current_state.current_token_index],
                        ]
                        + next_label_ids
                    ),
                    [
                        self.current_state.new_state(
                            "complete_word_outside"  # First token of a word, we need to complete the word
                            if self.next_is_subtoken()
                            else "next_outside",  # current word is complete, return to next_outside state
                            advance_current_token_index=True,
                            current_label_id=-1,
                        )
                    ]
                    + [
                        self.current_state.new_state(
                            "complete_label_start"
                            if self.start_labels_trie.next_is_last(label_id) == -1
                            else "complete_word_inside",
                            advance_current_token_index=False,
                            advance_current_label_token_index=True,
                            current_label_id=-1
                            if self.start_labels_trie.next_is_last(label_id) == -1
                            else self.start_labels_trie.next_is_last(label_id),
                        )
                        for label_id in next_label_ids
                    ],
                )
        elif self.current_state.state == "next_inside":
            """
            The model can choose between the next token or an closing the current tag.
            """
            # If the next token is a EOS token we close the current tag
            if (
                (
                    self.tokens_ids[self.current_state.current_token_index]
                    == self.eos_token_id
                    and self.current_state.current_token_index > 0
                )
                or self.tokens_ids[self.current_state.current_token_index]
                == self.pad_token_id
                or self.word_ids[self.current_state.current_token_index] is None
            ):
                trie = self.end_labels_dict[self.current_state.current_label_id]
                possible_end_labels = trie.get_next_labels()

                return (
                    torch.tensor(possible_end_labels),
                    [
                        self.current_state.new_state(
                            "complete_label_end"
                            if trie.next_is_last(label_id) == -1
                            else "next_outside",
                            advance_current_token_index=False,
                            advance_current_label_token_index=True,
                            current_label_id=self.current_state.current_label_id,
                        )
                        for label_id in possible_end_labels
                    ],
                )
            else:
                #  Choose between the next token or closing the current tag.
                trie = self.end_labels_dict[self.current_state.current_label_id]
                possible_end_labels = trie.get_next_labels()

                return (
                    torch.tensor(
                        [
                            self.tokens_ids[self.current_state.current_token_index]
                        ]  # Next token
                        + possible_end_labels  # Closing tag
                    ),
                    [
                        self.current_state.new_state(
                            "complete_word_inside"  # First token of a word, we need to complete the word
                            if self.next_is_subtoken()
                            else "next_inside",  # current word is complete, return to next_inside state
                            advance_current_token_index=True,
                            current_label_id=self.current_state.current_label_id,
                        )
                    ]
                    + [
                        self.current_state.new_state(
                            "complete_label_end"
                            if trie.next_is_last(label_id) == -1
                            else "next_outside",
                            advance_current_token_index=False,
                            advance_current_label_token_index=True,
                            current_label_id=self.current_state.current_label_id,
                        )
                        for label_id in possible_end_labels
                    ],
                )

        elif self.current_state.state == "complete_word_outside":
            """
            The model can only choose to complete current word (we are not inside a label).
            """

            return (
                torch.tensor([self.tokens_ids[self.current_state.current_token_index]]),
                [
                    self.current_state.new_state(
                        "complete_word_outside"  # Sub-token of a word, we need to complete the word
                        if self.next_is_subtoken()
                        else "next_outside",  # current word is complete, return to next_outside state
                        advance_current_token_index=True,
                        current_label_id=-1,
                    )
                ],
            )
        elif self.current_state.state == "complete_word_inside":
            """
            The model can only choose to complete current word (we are inside a label).
            """
            return (
                torch.tensor([self.tokens_ids[self.current_state.current_token_index]]),
                [
                    self.current_state.new_state(
                        "complete_word_inside"  # Sub-token of a word, we need to complete the word
                        if self.next_is_subtoken()
                        else "next_inside",  # current word is complete, return to next_inside state
                        advance_current_token_index=True,
                        current_label_id=self.current_state.current_label_id,
                    )
                ],
            )
        elif self.current_state.state == "complete_label_start":
            """
            The model can only choose to complete the current label (we are at the start of the label).
            """
            next_label_ids = self.start_labels_trie.get_next_labels()

            return (
                torch.tensor(next_label_ids),
                [
                    self.current_state.new_state(
                        "complete_label_start"
                        if self.start_labels_trie.next_is_last(label_id) == -1
                        else "complete_word_inside",
                        advance_current_token_index=False,
                        advance_current_label_token_index=True,
                        current_label_id=-1
                        if self.start_labels_trie.next_is_last(label_id) == -1
                        else self.start_labels_trie.next_is_last(label_id),
                    )
                    for label_id in next_label_ids
                ],
            )

        elif self.current_state.state == "complete_label_end":
            """
            The model can only choose to complete the current label (we are at the end of the label).
            """

            trie = self.end_labels_dict[self.current_state.current_label_id]
            possible_end_labels = trie.get_next_labels()

            return (
                torch.tensor(possible_end_labels),
                [
                    self.current_state.new_state(
                        "complete_label_end"
                        if trie.next_is_last(label_id) == -1
                        else "complete_word_outside",
                        advance_current_token_index=False,
                        advance_current_label_token_index=True,
                        current_label_id=self.current_state.current_label_id,
                    )
                    for label_id in possible_end_labels
                ],
            )

        else:
            raise ValueError(f"Unexpected state: {self.current_state.state}")

    def update(self, token_id, new_state):
        self.current_seq.append(token_id)
        self.current_state = new_state.copy()
        if new_state.state == "complete_label_start":
            self.start_labels_trie.move_to_label(token_id)
        else:
            self.start_labels_trie.reset()

        if new_state.state == "complete_label_end":
            self.end_labels_trie = self.end_labels_dict[new_state.current_label_id]
            self.end_labels_trie.move_to_label(token_id)
        else:
            for trie in self.end_labels_dict.values():
                trie.reset()

    def copy(self):
        new_constraint = SequenceLabellingConstraint(
            tokens_ids=self.tokens_ids,
            word_ids=self.word_ids,
            start_labels_ids=self.start_labels_ids,
            end_labels_ids=self.end_labels_ids,
            start_labels_names=self.start_labels_names,
            end_labels_names=self.end_labels_names,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        new_constraint.current_seq = self.current_seq.copy()
        new_constraint.state = self.current_state.copy()
        new_constraint.start_labels_trie = self.start_labels_trie.copy()
        new_constraint.end_labels_dict = {
            label_id: trie.copy() for label_id, trie in self.end_labels_dict.items()
        }
        return new_constraint

    def is_complete(self):
        return self.current_state.state == "end"

    def __len__(self):
        return self.current_state.seqlen

    def generated_tokens(self):
        return self.current_seq


class BeamNode:
    def __init__(
        self,
        decoder_context: List[int],
        tokens_ids: List[int],
        word_ids: List[int],
        start_labels_ids: List[List[int]],
        end_labels_ids: List[List[int]],
        start_labels_names: List[int],
        end_labels_names: List[int],
        pad_token_id: int,
        eos_token_id: int,
        device: torch.device = "cpu",
    ):
        self.log_prob = 0
        self.decoder_context = decoder_context
        self.seq = torch.tensor(decoder_context, dtype=torch.long, device=device)
        self.start_labels_ids = start_labels_ids
        self.end_labels_ids = end_labels_ids
        self.start_labels_names = start_labels_names
        self.end_labels_names = end_labels_names
        self.generation_tree = SequenceLabellingConstraint(
            tokens_ids=tokens_ids,
            word_ids=word_ids,
            start_labels_ids=start_labels_ids,
            end_labels_ids=end_labels_ids,
            start_labels_names=start_labels_names,
            end_labels_names=end_labels_names,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        self.completed = False
        self.device = device
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def report_state(self):
        print(
            f"=== BEAM NODE ===\n"
            f"log_prob: {self.log_prob}\n"
            f"completed {self.completed}"
        )
        self.generation_tree.report_state()

    @property
    def score(self):
        if len(self.generation_tree) == 0:
            raise ValueError(
                f"Cannot compute score of an empty sequence.\n"
                f"seq: {self.seq}\n"
                f"generated_tokens: {self.generation_tree.generated_tokens()}\n"
                f"log_prob: {self.log_prob}\n"
                f"completed {self.completed}\n"
            )
        return self.log_prob / len(self.generation_tree)

    def is_completed(self):
        return self.completed

    def copy(self):
        new_node = BeamNode(
            decoder_context=self.decoder_context,
            tokens_ids=self.generation_tree.tokens_ids,
            word_ids=self.generation_tree.word_ids,
            start_labels_ids=self.start_labels_ids,
            end_labels_ids=self.end_labels_ids,
            start_labels_names=self.start_labels_names,
            end_labels_names=self.end_labels_names,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            device=self.device,
        )
        new_node.log_prob = self.log_prob
        new_node.seq = self.seq.clone()
        new_node.generation_tree = self.generation_tree.copy()
        new_node.completed = self.completed
        return new_node

    def add_token(self, logits):
        new_nodes = []
        # Compute the next possible tokens
        possible_next_tokens, possible_next_states = self.generation_tree.advance()
        # Compute softmax for the possible tokens
        beam_logits = logits[possible_next_tokens].double().cpu()
        # print(
        #    [tokenizer_g.decode(x) for x in possible_next_tokens],
        #    logits[possible_next_tokens],
        # )
        for token_id, prob, next_state in zip(
            possible_next_tokens.tolist(), beam_logits, possible_next_states
        ):
            # print(f"token_id: {token_id}, prob: {prob}, next_state: {next_state}")
            # Create new node for each possible token
            new_node = self.copy()
            # Add token to sequence
            new_node.seq = torch.cat(
                [new_node.seq, torch.tensor([token_id]).to(self.device)]
            )
            # Compute score
            if (
                token_id == self.pad_token_id or self.completed
                # or token_id == self.eos_token_id
            ):
                # Ignore pad token
                new_node.log_prob = self.log_prob
            else:
                new_node.log_prob += float(torch.log(prob).item())
                # print(
                #    f"token_id: {token_id}, prob: {float(torch.log(prob).item())}, log_prob: {new_node.log_prob}"
                # )

            # Update constraint generation
            new_node.generation_tree.update(token_id=token_id, new_state=next_state)
            if next_state.state == "end":
                new_node.completed = True
            new_nodes.append(new_node)

        return new_nodes

    def get_decoder_context(self):
        return self.seq

    def get_generated_tokens(self):
        return self.generation_tree.generated_tokens()


class BeamSent:
    def __init__(
        self,
        decoder_context: List[int],
        tokens_ids: List[int],
        word_ids: List[int],
        start_labels_ids: List[List[int]],
        end_labels_ids: List[List[int]],
        start_labels_names: List[int],
        end_labels_names: List[int],
        pad_token_id: int,
        eos_token_id: int,
        device: torch.device = "cpu",
        num_beams: int = 1,
    ):
        self.decoder_context = decoder_context
        self.nodes = [
            BeamNode(
                decoder_context=decoder_context,
                tokens_ids=tokens_ids,
                word_ids=word_ids,
                start_labels_ids=start_labels_ids,
                end_labels_ids=end_labels_ids,
                start_labels_names=start_labels_names,
                end_labels_names=end_labels_names,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                device=device,
            )
            for _ in range(num_beams)
        ]

        # We start all the beams with the same token, this can produce duplicates
        # so we set the score of every beam except the first to -inf, so that
        # duplicates will not be selected as the topk candidates
        for node in self.nodes[1:]:
            node.log_prob = float("-inf")

        self.device = device
        self.num_beams = num_beams

    def report_state(self):
        print(f"=== BEAM SENT ===\n" f"num nodes: {len(self.nodes)}")

        for i, node in enumerate(self.nodes):
            print(f"=== NODE {i} ===")
            print(node.report_state())

        print("\n")

    def is_completed(self):
        return all([node.is_completed() for node in self.nodes])

    def update(self, logits) -> List[int]:
        beam_idx = []
        new_nodes = []
        for node_id, (node_logits, node) in enumerate(zip(logits, self.nodes)):
            ns = node.add_token(node_logits)
            new_nodes.extend(ns)
            beam_idx.extend([node_id] * len(ns))

        # for node in new_nodes:
        #    print(node.report_state())
        # print("\n")
        self.nodes = new_nodes

        # self.nodes = sorted(self.nodes, key=lambda x: x.score, reverse=True)[
        #    : self.num_beams
        # ]

        # Get idx of nodes with highest score
        sort_idx = torch.argsort(
            torch.tensor([node.score for node in self.nodes]), descending=True
        )[0 : self.num_beams]

        # Sort nodes by score
        self.nodes = [self.nodes[i] for i in sort_idx]

        return [beam_idx[i] for i in sort_idx]

    def get_decoder_contexts(self):
        return torch.vstack([node.get_decoder_context() for node in self.nodes])

    def get_generated_tokens(self):
        return torch.vstack(
            [torch.tensor(node.get_generated_tokens()) for node in self.nodes],
        ).to(self.device)

    def get_scores(self):
        return [node.score for node in self.nodes]


def run_model(
    model: Union[PreTrainedModel, torch.nn.Module],
    input_ids: torch.tensor,
    decoder_args: Dict,
    is_encoder_decoder: bool,
):
    # print(decoder_args.keys())

    gen_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids, **decoder_args
    )

    decoder_outputs = model(
        **gen_inputs,
        return_dict=True,
    )

    logits = decoder_outputs.logits
    logits = logits[:, -1, :]
    logits = torch.nn.functional.softmax(logits, dim=-1)

    decoder_args = model._update_model_kwargs_for_generation(
        decoder_outputs,
        decoder_args,
        is_encoder_decoder=is_encoder_decoder,
    )

    del decoder_outputs

    return logits, decoder_args


def constrained_beam_search(
    model_inputs: BatchEncoding,
    model: Union[PreTrainedModel, torch.nn.Module],
    start_labels_ids: List[List[int]],
    end_labels_ids: List[List[int]],
    start_labels_names: List[int],
    end_labels_names: List[int],
    pad_token_id: int,
    eos_token_id: int,
    max_length: int = 128,
    num_beams: int = 4,
    num_return_sequences: int = 1,
    forced_bos_token_id: int = None,
    use_cache: bool = True,
):
    if num_return_sequences < 0:
        raise ValueError(
            f"`num_return_sequences` must be >= 0, but is {num_return_sequences}"
        )
    if num_beams < 0:
        raise ValueError(f"`max_beams` must be >= 0, but is {num_beams}")
    if num_return_sequences > num_beams:
        raise ValueError(
            f"`num_return_sequences` must be <= `num_beams`, but is {num_return_sequences} and {num_beams}"
        )
    if max_length < 0:
        raise ValueError(f"`max_length` must be >= 0, but is {max_length}")

    with torch.no_grad():
        model.eval()
        sentence_beams = []

        if model.config.is_encoder_decoder:
            if forced_bos_token_id is None:
                decoder_initial_tokens = [
                    [model.generation_config.decoder_start_token_id]
                ] * len(model_inputs.input_ids)
            else:
                decoder_initial_tokens = [[forced_bos_token_id]] * len(
                    model_inputs.input_ids
                )
        else:
            if forced_bos_token_id is not None:
                print(
                    "Warning: forced_bos_token_id is ignored for non encoder-decoder models. If you want to add a "
                    "prompt use the '--prompt' flag. Run 'seq2seq.py --help' for more information."
                )
            decoder_initial_tokens = model_inputs.input_ids.cpu().tolist()

        for (
            encoder_inputs,
            sent_decoder_initial_tokens,
            sent_tokens,
            sent_word_ids,
        ) in zip(
            model_inputs.input_ids,
            decoder_initial_tokens,
            model_inputs.original_sentence_ids.cpu().tolist(),
            model_inputs.words_ids.cpu().tolist(),
        ):
            sentence_beams.append(
                BeamSent(
                    decoder_context=sent_decoder_initial_tokens,
                    tokens_ids=sent_tokens,
                    word_ids=sent_word_ids,
                    start_labels_ids=start_labels_ids,
                    end_labels_ids=end_labels_ids,
                    start_labels_names=start_labels_names,
                    end_labels_names=end_labels_names,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    device=model.device,
                    num_beams=num_beams,
                )
            )

        # Get the encoder output, we need to duplicate model_inputs for each beam

        encoder_inputs = model_inputs["input_ids"].repeat_interleave(
            repeats=num_beams, dim=0
        )
        encoder_mask = model_inputs["attention_mask"].repeat_interleave(
            repeats=num_beams, dim=0
        )

        kwargs = {
            "attention_mask": encoder_mask,
            "num_beams": num_beams,
            "decoder_start_token_id": model.generation_config.decoder_start_token_id,
            "do_sample": False,
            "temperature": None,
            "top_k": None,
            "top_p": None,
        }
        generation_config = copy.deepcopy(model.generation_config)
        generation_config, model_kwargs = model._prepare_generation_config(
            generation_config, **kwargs
        )
        model_kwargs["use_cache"] = use_cache
        #print(model_kwargs)

        generation_config.validate()
        model._validate_model_kwargs(model_kwargs.copy())

        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            encoder_inputs, model.generation_config.decoder_start_token_id, model_kwargs
        )

        model._prepare_special_tokens(
            generation_config,
            model_kwargs.get("attention_mask", None) is not None,
            device=inputs_tensor.device,
        )

        if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor,
                model_kwargs,
                model_input_name,
                generation_config=generation_config,
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if model.config.is_encoder_decoder:
            input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
                batch_size=8,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=torch.tensor(
                    generation_config.decoder_start_token_id
                ),
                # bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = (
                inputs_tensor
                if model_input_name == "input_ids"
                else model_kwargs.pop("input_ids")
            )

        if (
            generation_config.cache_implementation is None
            and model._supports_default_dynamic_cache()
        ):
            past = model_kwargs.get("past_key_values", None)
            requires_cross_attention_cache = (
                model.config.is_encoder_decoder
                or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs["past_key_values"] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                use_dynamic_cache_by_default = True
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)

        first = True
        while not all([sent.is_completed() for sent in sentence_beams]):
            # Get the input ids for the decoder.
            # print(f"len sentence beams: {len(sentence_beams)}")
            input_ids = torch.vstack(
                [sent.get_decoder_contexts() for sent in sentence_beams]
            ).to(model.device)
            # print(f"Loop inputs:")
            # print(input_ids.size())
            # print(input_ids)
            # Run the model
            logits, model_kwargs = run_model(
                model=model,
                input_ids=input_ids,
                decoder_args=model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )

            # Update the beams
            beam_idx = []
            for sent_no, sent in enumerate(sentence_beams):
                sent_idx = sent.update(
                    logits[sent_no * num_beams : (sent_no + 1) * num_beams]
                )
                sent_idx = [
                    i + (num_beams * sent_no) for i in sent_idx
                ]  # Adjust indices for each sentence in batch
                beam_idx.extend(sent_idx)

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = model._temporary_reorder_cache(
                    model_kwargs["past_key_values"], torch.tensor(beam_idx)
                )
            else:
                if first:
                    print(
                        f"Warning! model_kwargs['past_key_values'] is None, this means that we wont cache the past states. "
                        f"This will slow down the generation. This is not expected if use_cache=True. "
                        f"You have set use_cache={use_cache}. If you set use_cache=True and you still see this message "
                        f"it probably means that your model does not support caching."
                    )

            first = False

        return torch.vstack(
            [
                sent.get_generated_tokens()[:num_return_sequences]
                for sent in sentence_beams
            ]
        )


def unconstrained_beam_search(
    model_inputs: BatchEncoding,
    model: Union[PreTrainedModel, torch.nn.Module],
    max_length: int = 128,
    num_beams: int = 4,
    num_return_sequences: int = 1,
    forced_bos_token_id: int = None,
):
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "forced_bos_token_id": forced_bos_token_id,
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }
    with torch.no_grad():
        model.eval()
        generated_tokens = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            **gen_kwargs,
        )

        if model.config.is_encoder_decoder:
            return generated_tokens
        else:
            len_inputs = len(model_inputs.input_ids[0])
            return generated_tokens[:, len_inputs:]
