import os
import ast
import math
import datasets
import pandas as pd
from pprint import pprint

_CITATION = """
@inproceedings{sarti-etal-2022-divemt,
    title = "{D}iv{EMT}: Neural Machine Translation Post-Editing Effort Across Typologically Diverse Languages",
    author = "Sarti, Gabriele and Bisazza, Arianna and Guerberof Arenas, Ana and Toral, Antonio",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.532",
    pages = "7795--7816",
}
"""

_DESCRIPTION = """\
DivEMT is the first publicly available post-editing study of Neural Machine Translation (NMT) over a typologically diverse set of target languages. Using a strictly controlled setup, 18 professional translators were instructed to translate or post-edit the same set of English documents into Arabic, Dutch, Italian, Turkish, Ukrainian, and Vietnamese. During the process, their edits, keystrokes, editing times, pauses, and perceived effort were logged, enabling an in-depth, cross-lingual evaluation of NMT quality and its post-editing process.
"""

_HOMEPAGE = "https://github.com/gsarti/divemt"

_LICENSE = "GNU General Public License v3.0"

_ROOT_PATH = "https://huggingface.co/datasets/GroNLP/divemt/resolve/main"

_PATHS = {
    "main": os.path.join(_ROOT_PATH, "main.tsv"),
    "warmup": os.path.join(_ROOT_PATH, "warmup.tsv"),
}

_ALL_FIELDS = ['unit_id', 'flores_id', 'item_id', 'subject_id', 'lang_id', 'doc_id', 
     'task_type', 'translation_type', 'src_len_chr', 'mt_len_chr', 'tgt_len_chr',
     'src_len_wrd', 'mt_len_wrd', 'tgt_len_wrd', 'edit_time', 'k_total',
     'k_letter', 'k_digit', 'k_white', 'k_symbol', 'k_nav', 'k_erase',
     'k_copy', 'k_cut', 'k_paste', 'k_do', 'n_pause_geq_300',
     'len_pause_geq_300', 'n_pause_geq_1000', 'len_pause_geq_1000',
     'event_time', 'num_annotations', 'last_modification_time', 'n_insert',
     'n_delete', 'n_substitute', 'n_shift', 'tot_shifted_words', 'tot_edits',
     'hter', 'cer', 'bleu', 'chrf', 'time_s', 'time_m',
     'time_h', 'time_per_char', 'time_per_word', 'key_per_char',
     'words_per_hour', 'words_per_minute', 'per_subject_visit_order',
     'src_text', 'mt_text', 'tgt_text', 'aligned_edit', 'src_tokens', 'src_annotations',
     'mt_tokens', 'mt_annotations', 'tgt_tokens', 'tgt_annotations', 'src_wmt22_qe', 'mt_wmt22_qe'
]

_FLOAT_FIELDS = ["edit_time", "bleu", "chrf", "hter", "n_insert", "n_delete", "n_substitute",
    "n_shift", "time_s", "time_m", "time_h", 'time_per_char', 'time_per_word', 'key_per_char',
    'words_per_hour', 'words_per_minute', 'tot_shifted_words', 'tot_edits', "mt_len_chr",
    "mt_len_wrd", "cer"
]

_STR_FIELDS = ["unit_id", "item_id", "subject_id", "lang_id", "task_type", "translation_type",
    "src_text", "mt_text", "tgt_text", "aligned_edit"
]

_STR_SEQ_FIELDS = ['src_tokens', 'mt_tokens', 'tgt_tokens', 'src_wmt22_qe', 'mt_wmt22_qe']

_LANG_ANNOTATIONS_FIELDS = ['src_annotations', 'mt_annotations', 'tgt_annotations']


class DivEMTConfig(datasets.BuilderConfig):
    """BuilderConfig for the DivEMT Dataset."""

    def __init__(
        self,
        features,
        **kwargs,
    ):
        """
        Args:
        features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
        **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.features = features


class DivEMT(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DivEMTConfig(
            name=name,
            features=_ALL_FIELDS,
        )
        for name in ["warmup", "main"]
    ]

    DEFAULT_CONFIG_NAME = "main"

    def _info(self):
        features = {feature: datasets.Value("int32") for feature in self.config.features}
        for field in _ALL_FIELDS:
            if field in self.config.features:
                if field in _STR_FIELDS:
                    features[field] = datasets.Value("string")
                if field in _FLOAT_FIELDS:
                    features[field] = datasets.Value("float32")
                if field in _STR_SEQ_FIELDS:
                    features[field] = datasets.Sequence(datasets.Value("string"))
                if field in _LANG_ANNOTATIONS_FIELDS:
                    features[field] = datasets.features.Sequence(
                        {
                            "lemma": datasets.Value("string"),
                            "upos": datasets.Value("string"),
                            "feats": datasets.Value("string"),
                            "head": datasets.Value("string"),
                            "deprel": datasets.Value("string"),
                            "start_char": datasets.Value("int32"),
                            "end_char": datasets.Value("int32"),
                            "ner": datasets.Value("string"),
                        }
                    )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_PATHS[self.config.name])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir,
                    "features": self.config.features,
                },
            )
        ]
    
    def _generate_examples(self, filepath: str, features):
        """Yields examples as (key, example) tuples."""
        data = pd.read_csv(filepath, sep="\t")
        data = data[features]
        for id_, row in data.iterrows():
            row_dic = row.to_dict()
            for field in _STR_SEQ_FIELDS + _LANG_ANNOTATIONS_FIELDS:
                if isinstance(row_dic[field], float) and math.isnan(row_dic[field]):
                    row_dic[field] = []
                else:
                    row_dic[field] = ast.literal_eval(row_dic[field])
            yield id_, row_dic