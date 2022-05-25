from __future__ import absolute_import, division, print_function

import json
import logging
import os
import re
from collections import Counter

from src.utils.data_config import *
from src.utils.convqa_utils import normalize_answer

import spacy
import datasets

try:
    nlp = spacy.load('en_core_web_sm')
except:
    from spacy.cli.download import download
    download(model="en_core_web_sm")

MAX_Q_LEN = 100  # Max length of question
YOUR_LOCAL_DOWNLOAD = "data"  # For subtask1, Doc2Dial v1.0.1 is already included in the folder "data".

_CITATION = """\
    @article{reddy-etal-2019-coqa,
        title = "{C}o{QA}: A Conversational Question Answering Challenge",
        author = "Reddy, Siva  and
        Chen, Danqi  and
        Manning, Christopher D.",
        journal = "Transactions of the Association for Computational Linguistics",
        volume = "7",
        month = mar,
        year = "2019",
        url = "https://www.aclweb.org/anthology/Q19-1016",
        doi = "10.1162/tacl_a_00266",
        pages = "249--266",
        }
"""

_DESCRIPTION = """\
    CoQA is a large-scale dataset for building Conversational Question Answering systems. \
    The goal of the CoQA challenge is to measure the ability of machines to understand a text passage\
    and answer a series of interconnected questions that appear in a conversation. \
    CoQA is pronounced as coca.
"""

_HOMEPAGE = "https://stanfordnlp.github.io/coqa/"


_URLs = "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json, https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"


class Coqa(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coqa_rc",
            version=VERSION,
            description="Load CoQA dataset for machine reading comprehension tasks",
        ),
        datasets.BuilderConfig(
            name="coqa_ppo",
            version=VERSION,
            description="Load CoQA dataset for PPO training",
        ),
    ]

    DEFAULT_CONFIG_NAME = "coqa_rc"

    def _info(self):
        if self.config.name == "coqa_rc":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "span_text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            "answer_end": datasets.Value("int32"),
                            "rational_answer_start": datasets.Value("int32"),
                            "rational_answer_end": datasets.Value("int32"),
                        },
                    ),
                    "domain": datasets.Value("string"),
                    "answer_type": datasets.Value("string"),
                    "answer_option": datasets.Value("string"),
                }
            )
        elif self.config.name == "coqa_ppo":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "history": datasets.features.Sequence(datasets.Value("string")),
                    "orig_question": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "span_text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            "answer_end": datasets.Value("int32"),
                            "rational_answer_start": datasets.Value("int32"),
                            "rational_answer_end": datasets.Value("int32"),
                        },
                    ),
                    "domain": datasets.Value("string"),
                    "answer_type": datasets.Value("string"),
                    "answer_option": datasets.Value("string"),
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
        data_dir = YOUR_LOCAL_DOWNLOAD  # point to local dir to avoid downloading the dataset again
        return [datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "coqa/coqa-dev-v1.0.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "coqa/valid_split.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "coqa/train_split.json"
                        ),
                    },
                ),]
    
    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

    def space_extend(self, matchobj):
        return ' ' + matchobj.group(0) + ' '


    def pre_proc(self, text):
        text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', self.space_extend, text)
        text = text.strip(' \n')
        text = re.sub('\s+', ' ', text)
        return text

    def process(self, parsed_text):
        output = {'word': [], 'offsets': [], 'sentences': []}

        for token in parsed_text:
            output['word'].append(self._str(token.text))
            output['offsets'].append((token.idx, token.idx + len(token.text)))

        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        return output

    def get_raw_context_offsets(self, words, raw_text):
        raw_context_offsets = []
        p = 0
        for token in words:
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                print('something is wrong! token', token, 'raw_text:', raw_text)

            raw_context_offsets.append((p, p + len(token)))
            p += len(token)

        return raw_context_offsets
    
    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        if start_index == end_index == -1:
            return (-1, -1)
        return (offsets[start_index][0], offsets[end_index][1])

    
    def find_span_with_gt(self, context, offsets, ground_truth):
        best_f1 = 0.0
        best_span = (-1, -1)
        gt = normalize_answer(self.pre_proc(ground_truth)).split()

        ls = [
            i for i in range(len(offsets))
            if context[offsets[i][0]:offsets[i][1]].lower() in gt
        ]

        for i in range(len(ls)):
            for j in range(i, len(ls)):
                pred = normalize_answer(
                    self.pre_proc(
                        context[offsets[ls[i]][0]:offsets[ls[j]][1]])).split()
                common = Counter(pred) & Counter(gt)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred)
                    recall = 1.0 * num_same / len(gt)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_span = (offsets[ls[i]][0], offsets[ls[j]][1])
        return best_span


    def _generate_example(self, context, raw_context_offsets, input_answer, orig_start, orig_end, answer_type):
        if answer_type == "unknown":
            return -1, -1, -1, -1
        
        context_offsets = [offset for offset in raw_context_offsets if not (offset[1]<orig_start or offset[0]>orig_end)]
        start, end = orig_start, orig_end
        span_text = context[orig_start:orig_end].lower()
        
        while len(span_text) > 0 and self.is_whitespace(span_text[0]):
            span_text = span_text[1:]
            start += 1
        while len(span_text) > 0 and self.is_whitespace(span_text[-1]):
            span_text = span_text[:-1]
            end -= 1
        r_start, r_end = self.find_span(raw_context_offsets, start, end)

        if answer_type == "yes" or answer_type == "no":
            return -1, -1, r_start, r_end

        # for answer type as span
        input_text = input_answer.strip().lower()
        if input_text in span_text:
            p = span_text.find(input_text)
            answer_span = self.find_span(context_offsets,
                                         start + p,
                                         start + p + len(input_text))
        else:
            answer_span = self.find_span_with_gt(context, context_offsets, input_text)
            if answer_span[0] == answer_span[1] == -1:
                answer_span = self.find_span_with_gt(context, raw_context_offsets, input_text)
        
        answer_start, answer_end = answer_span[0], answer_span[1]

        return answer_start, answer_end, r_start, r_end
    
    def _make_additonal_answers(self, answers_list, turn_id):
        format_answers = []
        for answers in answers_list:
            answer = answers[turn_id]
            format_answers.append({
                "text": answer["input_text"],
                "span_text": answer["span_text"], 
                "answer_start": 0,
                "answer_end": 0, 
                "rational_answer_start": answer["span_start"],
                "rational_answer_end": answer["span_end"],
            })
        return format_answers


    def _get_answer_type(self,
                         question,
                         answer):
        if answer is not None:
            norm_answer = normalize_answer(answer["input_text"])
            if norm_answer in ["yes", "yese", "ye", "es"]:
                norm_answer = "yes"
            if norm_answer in ["no", "no not at all", "not", "not at all", "not yet", "not really"]:
                norm_answer = "no"
            
            if norm_answer == "unknown" or "bad_turn" in answer:
                return "unknown", None
            
            if norm_answer == "yes":
                return "yes", None
            
            if norm_answer == "no":
                return "no", None
            
            if norm_answer in ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]:
                return "number", norm_answer
        else:
            norm_answer = None
        
        norm_question_tokens = normalize_answer(question["input_text"]).split(" ")
        if "or" in norm_question_tokens:
            index = norm_question_tokens.index("or")
            if index-1 >= 0 and index+1 < len(norm_question_tokens):
                if norm_question_tokens[index-1] in QWORDS or norm_question_tokens[index+1] in QWORDS:
                    return "span", None

                if norm_answer == norm_question_tokens[index-1]:
                    norm_answer = "option_a"
                    return norm_answer, "|||".join([norm_question_tokens[index-1], norm_question_tokens[index+1]])
                elif norm_answer == norm_question_tokens[index+1]:
                    norm_answer = "option_b"
                    return norm_answer, "|||".join([norm_question_tokens[index-1], norm_question_tokens[index+1]])
                else:
                    return "option", "|||".join([norm_question_tokens[index-1], norm_question_tokens[index+1]])
        
        return "span", None
    
    def _get_questions(self, all_prev_utterances, current_only=False):
        if current_only:
            question = all_prev_utterances[-1]
        else:
            question_str = " ".join(
                        list(reversed(all_prev_utterances))
                    ).strip()
            question = " ".join(question_str.split()[:MAX_Q_LEN])
        return question

    def _generate_examples(self, filepath):
        if self.config.name == "coqa_rc":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.info("generating examples from = %s", filepath)
            with open(filepath, "r") as f:
                data = json.load(f)["data"]
            
            nlp = spacy.load('en_core_web_sm', parser=False)
            for row in data:
                context = row["story"]
                annotated_context = self.process(nlp(self.pre_proc(context)))
                raw_context_offsets = self.get_raw_context_offsets(annotated_context['word'], context)

                domain = row["source"]
                title = row["filename"]
                all_prev_utterances = []

                if "additional_answers" in row:
                    additional_answers = row["additional_answers"].values()
                else:
                    additional_answers = None

                for i, (question, answer) in enumerate(zip(row['questions'], row["answers"])):
                    id_ = str(row['id']) + '_' + str(question['turn_id'])
                    all_prev_utterances.append(question['input_text'])
                    orig_answer_text = answer["input_text"]
                    orig_answer_start = answer["span_start"]
                    orig_answer_end = answer["span_end"]

                    answer_type, answer_subtype = self._get_answer_type(question, answer)
                    if answer_type == "yes":
                        orig_answer_text = 'yes'
                    elif answer_type == "no":
                        orig_answer_text = 'no'
                    elif answer_type == 'unknown':
                        orig_answer_text = 'unknown'
                    
                    if "option" in answer_type:
                        options = answer_subtype.split("|||")
                        for option in options:
                            for word in annotated_context['word']:
                                if option == word.lower():
                                    answer_type = "span"
                                    answer_subtype = None
                                    break

                    answer_start, answer_end, r_start, r_end = self._generate_example(
                        context, 
                        raw_context_offsets, 
                        orig_answer_text, 
                        orig_answer_start, 
                        orig_answer_end,
                        answer_type
                    )

                    answers =  [{
                        "text": orig_answer_text,
                        "span_text": answer["span_text"], 
                        "answer_start": answer_start, # orig_answer_start, # 
                        "answer_end": answer_end, # orig_answer_end, #
                        "rational_answer_start": r_start,
                        "rational_answer_end": r_end,
                    }]

                    if additional_answers is not None:
                        answers += self._make_additonal_answers(additional_answers, i)

                    question = self._get_questions(all_prev_utterances, current_only=("only" in self.config.name))
                        
                    # append the original answer into the utterance list
                    all_prev_utterances.append(orig_answer_text)

                    qa = {
                            "id": id_,
                            "domain": domain,
                            "title": title,
                            "context": context,
                            "question": question,
                            "answers": answers,
                            "answer_type": answer_type, 
                            "answer_option": "" if answer_subtype is None else answer_subtype,
                    }
                    yield id_, qa

        elif self.config.name == "coqa_ppo":
            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.info("generating examples from = %s", filepath)
            with open(filepath, "r") as f:
                data = json.load(f)["data"]
            
            nlp = spacy.load('en_core_web_sm') #, parser=False)
            for row in data:
                context = row["story"]
                annotated_context = self.process(nlp(self.pre_proc(context)))
                raw_context_offsets = self.get_raw_context_offsets(annotated_context['word'], context)

                domain = row["source"]
                title = row["filename"]
                all_prev_utterances = []

                if "additional_answers" in row:
                    additional_answers = row["additional_answers"].values()
                else:
                    additional_answers = None

                for i, (question, answer) in enumerate(zip(row['questions'], row["answers"])):
                    id_ = str(row['id']) + '_' + str(question['turn_id'])
                    all_prev_utterances.append(question['input_text'])
                    orig_question = question['input_text']
                    orig_answer_text = answer["input_text"]
                    orig_answer_start = answer["span_start"]
                    orig_answer_end = answer["span_end"]

                    answer_type, answer_subtype = self._get_answer_type(question, answer)
                    if answer_type == "yes":
                        orig_answer_text = 'yes'
                    elif answer_type == "no":
                        orig_answer_text = 'no'
                    elif answer_type == 'unknown':
                        orig_answer_text = 'unknown'
                    
                    if "option" in answer_type:
                        options = answer_subtype.split("|||")
                        for option in options:
                            for word in annotated_context['word']:
                                if option == word.lower():
                                    answer_type = "span"
                                    answer_subtype = None
                                    break

                    answer_start, answer_end, r_start, r_end = self._generate_example(
                        context, 
                        raw_context_offsets, 
                        orig_answer_text, 
                        orig_answer_start, 
                        orig_answer_end,
                        answer_type
                    )

                    answers =  [{
                        "text": orig_answer_text,
                        "span_text": answer["span_text"], 
                        "answer_start": answer_start, # orig_answer_start, # 
                        "answer_end": answer_end, # orig_answer_end, #
                        "rational_answer_start": r_start,
                        "rational_answer_end": r_end,
                    }]

                    if additional_answers is not None:
                        answers += self._make_additonal_answers(additional_answers, i)

                    question = self._get_questions(all_prev_utterances, current_only=("only" in self.config.name))
                        
                    # append the original answer into the utterance list
                    all_prev_utterances.append(orig_answer_text)

                    qa = {
                            "id": id_,
                            "domain": domain,
                            "title": title,
                            "context": context,
                            "question": question,
                            "answers": answers,
                            "answer_type": answer_type, 
                            "answer_option": "" if answer_subtype is None else answer_subtype,
                            "history": all_prev_utterances[:-2],
                            "orig_question": orig_question,
                    }
                    yield id_, qa