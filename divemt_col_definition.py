"""
Column reference for the DivEMT dataset.

Definitions summarised from the dataset card:
https://huggingface.co/datasets/GroNLP/divemt

Usage
-----
    from divemt_col_definition import col_table, describe, COLUMN_DEFS

    col_table            # the whole table as a DataFrame
    describe("hter")     # the meaning of a single column
    COLUMN_DEFS["hter"]  # same thing, straight from the dict
"""

import pandas as pd

# The mapping: column name -> what it means.
# This is the part that has to be written by hand from the README,
# because the meanings don't exist anywhere inside the data itself.
COLUMN_DEFS = {
    "unit_id": "Full entry id, format flores101-{config}-{lang}-{doc_id}-{modality}-{sent_num}.",
    "flores_id": "Index of the sentence in the original Flores-101 dataset.",
    "item_id": "Sentence id; leading digits = document, final digit = position in that document.",
    "subject_id": "Identifier of the translator doing the task (e.g. vie_t1, vie_t2, vie_t3).",
    "lang_id": "Language of the sentence, Flores three-letter code (ara, nld, ita, tur, ukr, vie).",
    "doc_id": "Document identifier for the sentence.",
    "task_type": "Task modality: ht (from scratch), pe1 (post-edit Google Translate), pe2 (post-edit mBART).",
    "translation_type": "ht for translation from scratch, pe for post-editing.",
    "src_len_chr": "Length of the English source in characters.",
    "mt_len_chr": "Length of the machine translation in characters (NaN when task_type is ht).",
    "tgt_len_chr": "Length of the final target text in characters.",
    "src_len_wrd": "Length of the English source in words.",
    "mt_len_wrd": "Length of the machine translation in words (NaN when task_type is ht).",
    "tgt_len_wrd": "Length of the final target text in words.",
    "edit_time": "Total editing time for the sentence, in seconds.",
    "k_total": "Total number of keystrokes.",
    "k_letter": "Number of letter keystrokes.",
    "k_digit": "Number of digit keystrokes.",
    "k_white": "Number of whitespace keystrokes.",
    "k_symbol": "Number of symbol/punctuation keystrokes.",
    "k_nav": "Number of navigation keystrokes (arrow keys, mouse clicks).",
    "k_erase": "Number of erase keystrokes (backspace, delete).",
    "k_copy": "Number of copy (Ctrl+C) actions.",
    "k_cut": "Number of cut (Ctrl+X) actions.",
    "k_paste": "Number of paste (Ctrl+V) actions.",
    "k_do": "Number of Enter actions.",
    "n_pause_geq_300": "Number of pauses of 300 ms or more.",
    "len_pause_geq_300": "Total duration of pauses of 300 ms or more, in milliseconds.",
    "n_pause_geq_1000": "Number of pauses of 1 s or more.",
    "len_pause_geq_1000": "Total duration of pauses of 1 s or more, in milliseconds.",
    "event_time": "Total time summed across all translation events (usually close to edit_time).",
    "num_annotations": "How many times the translator focused the textbox (1 = done once, never revised).",
    "n_insert": "Post-editing insertions, via tercom (empty for ht).",
    "n_delete": "Post-editing deletions, via tercom (empty for ht).",
    "n_substitute": "Post-editing substitutions, via tercom (empty for ht).",
    "n_shift": "Post-editing shifts, via tercom (empty for ht).",
    "tot_shifted_words": "Total number of words moved across all shifts.",
    "tot_edits": "Sum of all edit types for the sentence.",
    "hter": "Human-mediated Translation Edit Rate between MT and the post-edited target (empty for ht).",
    "cer": "Character-level HTER (CharacTER) between MT and post-edited target (empty for ht).",
    "bleu": "Sentence-level BLEU between MT and post-edited target, via SacreBLEU (empty for ht).",
    "chrf": "Sentence-level chrF between MT and post-edited target, via SacreBLEU (empty for ht).",
    "time_s": "Edit time in seconds.",
    "time_m": "Edit time in minutes.",
    "time_h": "Edit time in hours.",
    "time_per_char": "Edit time per source character, in seconds.",
    "time_per_word": "Edit time per source word, in seconds.",
    "key_per_char": "Ratio of keystrokes to characters needed for the translation.",
    "words_per_hour": "Source words translated or post-edited per hour.",
    "words_per_minute": "Source words translated or post-edited per minute.",
    "per_subject_visit_order": "Order in which the translator opened documents (1 = first accessed).",
    "src_text": "The original English source sentence.",
    "mt_text": "The machine translation before post-editing (missing when task_type is ht).",
    "tgt_text": "The final sentence produced by the translator.",
    "aligned_edit": "Aligned REF / HYP / edit-operation view (I=insert, D=delete, S=substitute).",
    "src_tokens": "Source tokens from tokenising src_text with Stanza.",
    "src_annotations": "Per-token linguistic annotations (POS, NER, etc.) for the source, from Stanza.",
    "mt_tokens": "MT tokens from tokenising mt_text with Stanza.",
    "mt_annotations": "Per-token linguistic annotations for the MT, from Stanza.",
    "tgt_tokens": "Target tokens from tokenising tgt_text with Stanza.",
    "tgt_annotations": "Per-token linguistic annotations for the target, from Stanza.",
    # Added in the Feb 2023 update (verify these appear in your file with the check below):
    "src_wmt22_qe": "Word-level quality-estimation tags on the source, WMT22 QE Task 2 style.",
    "mt_wmt22_qe": "Word-level quality-estimation tags on the MT, WMT22 QE Task 2 style.",
}

# The dictionary becomes a two-column table in one line.
col_table = pd.DataFrame(
    list(COLUMN_DEFS.items()),
    columns=["column", "definition"],
)


def describe(name):
    """Return the plain-language meaning of a single column."""
    return COLUMN_DEFS.get(name, f"No definition on file for '{name}'.")