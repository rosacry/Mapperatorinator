import numpy.typing as npt
import torch

from ..tokenizer import Event, EventType, Tokenizer


def create_sequences(
        events: list[Event],
        seq_len: int,
) -> list[dict[str, int | npt.NDArray | list[Event]]]:
    """Create frame and token sequences for training/testing.

    Args:
        events: Events and time shifts.
        seq_len: The length of the sequence.

    Returns:
        A list of source and target sequences.
    """

    sequences = []
    timed_events = [EventType.CIRCLE, EventType.SLIDER_HEAD]

    for i in range(seq_len + 1, len(events)):
        event = events[i]

        if event.type not in timed_events:
            continue

        # Create the sequence
        sequence = {
            "events": events[i - 1 - seq_len:i - 1],
            "labels_event": events[i - 1],
        }

        sequences.append(sequence)

    return sequences


def tokenize_sequence(sequence: dict, tokenizer: Tokenizer) -> dict:
    """Tokenize the event sequence.

    Begin token sequence with `[SOS]` token (start-of-sequence).
    End token sequence with `[EOS]` token (end-of-sequence).

    Args:
        sequence: The input sequence.
        tokenizer: The tokenizer to use.

    Returns:
        The same sequence with tokenized events.
    """
    tokens = torch.empty(len(sequence["events"]), dtype=torch.long)
    for i, event in enumerate(sequence["events"]):
        tokens[i] = tokenizer.encode(event)
    sequence["input_ids"] = tokens
    del sequence["events"]

    labels = torch.empty(1, dtype=torch.long)
    labels[0] = tokenizer.encode(sequence["labels_event"])
    sequence["labels"] = labels
    del sequence["labels_event"]

    return sequence
