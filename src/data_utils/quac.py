from __future__ import absolute_import, division, print_function

import json
import logging
import os

import datasets

MAX_Q_LEN = 100  # Max length of question
YOUR_LOCAL_DOWNLOAD = "data"  # For subtask1, Doc2Dial v1.0.1 is already included in the folder "data".

_CITATION = """\
    @inproceedings{choi2018quac,
    title={QuAC: Question Answering in Context},
    author={Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
    pages={2174--2184},
    year={2018}
    }
"""

_DESCRIPTION = """\
    QuAC
"""

_HOMEPAGE = "https://quac.ai/"


_URLs = "https://s3.amazonaws.com/my89public/quac/train_v0.2.json, https://s3.amazonaws.com/my89public/quac/val_v0.2.json"

class Quac(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.2.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="quac_rc",
            version=VERSION,
            description="Load QuAC dataset for machine reading comprehension tasks",
        ),
        datasets.BuilderConfig(
            name="quac_ppo",
            version=VERSION,
            description="Load QuAC dataset for PPO training",
        ),
    ]

    DEFAULT_CONFIG_NAME = "quac_rc"

    def _info(self):

        if self.config.name == "quac_rc":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "question": datasets.Value("string"),
                    "no_answer": datasets.Value("bool"),
                    "yesno": datasets.Value("int32"),
                    "followup": datasets.Value("int32"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            "answer_end": datasets.Value("int32"),
                        }
                    ),
                    "domain": datasets.Value("string"),
                }
            )
        elif self.config.name == "quac_ppo":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "history": datasets.features.Sequence(datasets.Value("string")),
                    "orig_question": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "no_answer": datasets.Value("bool"),
                    "yesno": datasets.Value("int32"),
                    "followup": datasets.Value("int32"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            "answer_end": datasets.Value("int32"),
                        }
                    ),
                    "domain": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):

        my_urls = _URLs

        # data_dir = dl_manager.download_and_extract(my_urls) 
        data_dir = YOUR_LOCAL_DOWNLOAD # point to local dir to avoid downloading the dataset again

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "quac/val_v0.2.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "quac/valid.json"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "quac/train_clean.json"
                    ),
                },
            ),
        ]

    def _is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    
    def _get_answers(self, answers, char_to_word_offset, no_answer):
        # if no_answer:
        #     format_answers = [{"text": "CANNOT", "answer_start": -1, "answer_end": -1}]
        # else:
        answer = answers[0]
        answer_start = char_to_word_offset[answer["answer_start"]]
        answer_end = char_to_word_offset[min(answer["answer_start"] + len(answer["text"]) - 1, len(char_to_word_offset) - 1)]
        format_answers = [{
            "text": answer["text"], 
            "answer_start": answer_start, 
            "answer_end": answer_end,
        }]
        for answer in answers[1:]:
            format_answers.append(
                {
                "text": answer["text"], 
                "answer_start": answer["answer_start"], 
                "answer_end": answer["answer_start"]+len(answer["text"])
                }
            )
        return format_answers
    
    def _get_questions(self, all_prev_utterances, current_only=False):
        if current_only:
            question = all_prev_utterances[-1]
        else:
            question_str = " ".join(
                        list(reversed(all_prev_utterances[-3:]))
                    ).strip()
            question = " ".join(question_str.split())
        return question


    def _generate_examples(self, filepath):
        Alpha2Int = {"y":0, "n":2, "x": 1, "m": 1} # project alphabet to numbers, yes->0, no->2, neither/maybe->1

        if self.config.name == "quac_rc":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.info("generating examples from = %s", filepath)
            with open(filepath, "r") as f:
                data = json.load(f)["data"]
            
            for group in data:
                domain = group["title"]
                for item in group['paragraphs']:
                    title = item["id"]
                    context = item["context"]

                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True

                    # Split on whitespace so that different tokens may be attributed to their original position.
                    for c in context:
                        if self._is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    all_prev_utterances = []
                    for qa in item["qas"]:
                        id_ = qa["id"]
                        all_prev_utterances.append(qa["question"])

                        question = self._get_questions(all_prev_utterances, current_only=("only" in self.config.name))

                        # append the original answer into the utterance list
                        orig_answer_text = qa["orig_answer"]["text"]
                        no_answer = (orig_answer_text == "CANNOTANSWER")
                        yesno = Alpha2Int[qa["yesno"]]
                        followup = Alpha2Int[qa["followup"]]
                        
                        answers = self._get_answers(qa["answers"], char_to_word_offset, no_answer)

                        if not no_answer:
                            all_prev_utterances.append(orig_answer_text)
                        else:
                            all_prev_utterances.pop()

                        qa = {
                                "id": id_, # For subtask1, the id should be this format.
                                "title": title,
                                "context": doc_tokens,
                                "question": question,
                                "no_answer": no_answer,
                                "yesno": yesno,
                                "followup": followup,
                                "answers": answers,
                                "domain": domain,
                            }
                        yield id_, qa
            
        elif self.config.name == "quac_ppo":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.info("generating examples from = %s", filepath)
            with open(filepath, "r") as f:
                data = json.load(f)["data"]

            for group in data:
                domain = group["title"]
                for item in group['paragraphs']:
                    title = item["id"]
                    context = item["context"]

                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True

                    # Split on whitespace so that different tokens may be attributed to their original position.
                    for c in context:
                        if self._is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    all_prev_utterances = []
                    for qa in item["qas"]:
                        id_ = qa["id"]
                        all_prev_utterances.append(qa["question"])

                        question = self._get_questions(all_prev_utterances, current_only=("only" in self.config.name))

                        # append the original answer into the utterance list
                        orig_answer_text = qa["orig_answer"]["text"]
                        no_answer = (orig_answer_text == "CANNOTANSWER")
                        yesno = Alpha2Int[qa["yesno"]]
                        followup = Alpha2Int[qa["followup"]]
                        
                        answers = self._get_answers(qa["answers"], char_to_word_offset, no_answer)

                        if not no_answer:
                            all_prev_utterances.append(orig_answer_text)
                        else:
                            all_prev_utterances.pop()

                        _qa = {
                                "id": id_, 
                                "title": title,
                                "context": doc_tokens,
                                "question": question,
                                "no_answer": no_answer,
                                "yesno": yesno,
                                "followup": followup,
                                "answers": answers,
                                "domain": domain,
                                "history": all_prev_utterances[:-2] if not no_answer else all_prev_utterances,
                                "orig_question": qa["question"],
                            }
                        yield id_, _qa
                            