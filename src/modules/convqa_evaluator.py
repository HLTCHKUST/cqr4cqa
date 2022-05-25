"""Official evaluation script for CoQA.

The code is based partially on SQuAD 2.0 evaluation script.
"""
import argparse
import json
import sys
from collections import Counter, OrderedDict, defaultdict

from src.utils.convqa_utils import normalize_answer
OPTS = None

out_domain = ["reddit", "science"]
in_domain = ["mctest", "gutenberg", "race", "cnn", "wikipedia"]
domain_mappings = {"mctest":"children_stories", "gutenberg":"literature", "race":"mid-high_school", "cnn":"news", "wikipedia":"wikipedia", "science":"science", "reddit":"reddit"}


class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return normalize_answer(s).split()

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = Evaluator.get_tokens(a_gold)
        pred_toks = Evaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def preds_to_rows(pred_file):
        preds = json.load(open(pred_file))
        pred_lines = []
        for ids, pred in preds.items():
            pred_lines.append({
                "id": ids,
                "prediction_text": pred
            })
        return pred_lines
    
    @staticmethod
    def _compute_turn_score(a_gold_list, a_pred):
        f1_sum = 0.0
        em_sum = 0.0
        if len(a_gold_list) > 1:
            for i in range(len(a_gold_list)):
                # exclude the current answer
                gold_answers = a_gold_list[0:i] + a_gold_list[i + 1:]
                em_sum += max(Evaluator.compute_exact(a, a_pred) for a in gold_answers)
                f1_sum += max(Evaluator.compute_f1(a, a_pred) for a in gold_answers)
        else:
            em_sum += max(Evaluator.compute_exact(a, a_pred) for a in a_gold_list)
            f1_sum += max(Evaluator.compute_f1(a, a_pred) for a in a_gold_list)

        return {'em': em_sum / max(1, len(a_gold_list)), 'f1': f1_sum / max(1, len(a_gold_list))}


class CoQAEvaluator(Evaluator):

    def __init__(self):
        pass
    
    def gold_answers_to_dict_from_file(self, gold_file):
        dataset = json.load(open(gold_file))
        gold_dict = {}
        id_to_source = {}
        for story in dataset['data']:
            source = story['source']
            story_id = story['id']
            id_to_source[story_id] = source
            questions = story['questions']
            multiple_answers = [story['answers']]
            if 'additional_answers' in story:
                multiple_answers += story['additional_answers'].values()
            for i, qa in enumerate(questions):
                qid = qa['turn_id']
                if i + 1 != qid:
                    sys.stderr.write("Turn id should match index {}: {}\n".format(i + 1, qa))
                gold_answers = []
                for answers in multiple_answers:
                    answer = answers[i]
                    if qid != answer['turn_id']:
                        sys.stderr.write("Question turn id does match answer: {} {}\n".format(qa, answer))
                    gold_answers.append(answer['input_text'])
                key = (story_id, qid)
                if key in gold_dict:
                    sys.stderr.write("Gold file has duplicate stories: {}".format(source))
                gold_dict[key] = gold_answers
        return gold_dict, id_to_source
    
    def gold_answers_to_dict_from_data(self, ref_data):
        gold_dict = {}
        id_to_source = {}
        for item in ref_data:
            story_id, turn_id = self._parse_pred_key(item["id"])
            gold_answers = item["answers"]["text"]
            source = item["domain"]

            key = (story_id, turn_id)
            gold_dict[key] = gold_answers
            if story_id not in id_to_source:
                id_to_source[story_id] = source
        return gold_dict, id_to_source

    def _parse_pred_key(self, pred_key):
        splits = pred_key.split("_")
        turn_id = splits[-1]
        story_id = "_".join(splits[:-1])
        return str(story_id), int(turn_id)
    
    def _get_pred_key(self, story_id, turn_id):
        pred_key = story_id + "_" + str(turn_id)
        return pred_key
    
    def compute_turn_score(self, story_id, turn_id, a_pred, gold_data):
        ''' This is the function what you are probably looking for. a_pred is the answer string your model predicted. '''
        key = (story_id, turn_id)
        a_gold_list = gold_data[key]
        return CoQAEvaluator._compute_turn_score(a_gold_list, a_pred)


    def get_raw_scores(self, pred_data, gold_data):
        ''''Returns a dict with score with each turn prediction'''
        exact_scores = {}
        f1_scores = {}
        for idx, row in enumerate(pred_data):
            key, a_pred = row["id"], row["prediction_text"]
            story_id, turn_id = self._parse_pred_key(key)
            gold_key = (story_id, turn_id)
            if gold_key not in gold_data:
                sys.stderr.write(f'Missing reference for {key}\n')
            scores = self.compute_turn_score(story_id, turn_id, a_pred, gold_data)
            exact_scores[gold_key] = scores['em']
            f1_scores[gold_key] = scores['f1']

        return exact_scores, f1_scores

    def get_raw_scores_human(self, gold_data):
        ''''Returns a dict with score for each turn'''
        exact_scores = {}
        f1_scores = {}
        for story_id, turn_id in gold_data:
            key = (story_id, turn_id)
            f1_sum = 0.0
            em_sum = 0.0
            if len(gold_data[key]) > 1:
                for i in range(len(gold_data[key])):
                    # exclude the current answer
                    gold_answers = gold_data[key][0:i] + gold_data[key][i + 1:]
                    em_sum += max(Evaluator.compute_exact(a, gold_data[key][i]) for a in gold_answers)
                    f1_sum += max(Evaluator.compute_f1(a, gold_data[key][i]) for a in gold_answers)
            else:
                exit("Gold answers should be multiple: {}={}".format(key, gold_data[key]))
            exact_scores[key] = em_sum / len(gold_data[key])
            f1_scores[key] = f1_sum / len(gold_data[key])
        return exact_scores, f1_scores

    def human_performance(self, gold_data, id_to_source):
        exact_scores, f1_scores = self.get_raw_scores_human(gold_data)
        return self.get_domain_scores(exact_scores, f1_scores, gold_data, id_to_source)

    def model_performance(self, pred_data, gold_data, id_to_source):
        exact_scores, f1_scores = self.get_raw_scores(pred_data, gold_data)
        return self.get_domain_scores(exact_scores, f1_scores, gold_data, id_to_source)

    def get_domain_scores(self, exact_scores, f1_scores, gold_data, id_to_source):
        scores = OrderedDict()
        sources = {}
        for source in in_domain + out_domain:
            sources[source] = Counter()

        for story_id, turn_id in gold_data:
            key = (story_id, turn_id)
            source = id_to_source[story_id]
            sources[source]['em_total'] += exact_scores.get(key, 0)
            sources[source]['f1_total'] += f1_scores.get(key, 0)
            sources[source]['turn_count'] += 1

        in_domain_em_total = 0.0
        in_domain_f1_total = 0.0
        in_domain_turn_count = 0

        out_domain_em_total = 0.0
        out_domain_f1_total = 0.0
        out_domain_turn_count = 0

        for source in in_domain + out_domain:
            domain = domain_mappings[source]
            scores[domain] = {}
            scores[domain]['em'] = round(sources[source]['em_total'] / max(1, sources[source]['turn_count']) * 100, 2)
            scores[domain]['f1'] = round(sources[source]['f1_total'] / max(1, sources[source]['turn_count']) * 100, 2)
            scores[domain]['turns'] = sources[source]['turn_count']
            if source in in_domain:
                in_domain_em_total += sources[source]['em_total']
                in_domain_f1_total += sources[source]['f1_total']
                in_domain_turn_count += sources[source]['turn_count']
            elif source in out_domain:
                out_domain_em_total += sources[source]['em_total']
                out_domain_f1_total += sources[source]['f1_total']
                out_domain_turn_count += sources[source]['turn_count']

        scores["in_domain"] = {'em': round(in_domain_em_total / max(1, in_domain_turn_count) * 100, 2),
                            'f1': round(in_domain_f1_total / max(1, in_domain_turn_count) * 100, 2),
                            'turns': in_domain_turn_count}
        scores["out_domain"] = {'em': round(out_domain_em_total / max(1, out_domain_turn_count) * 100, 2),
                                'f1': round(out_domain_f1_total / max(1, out_domain_turn_count) * 100, 2),
                                'turns': out_domain_turn_count}

        em_total = in_domain_em_total + out_domain_em_total
        f1_total = in_domain_f1_total + out_domain_f1_total
        turn_count = in_domain_turn_count + out_domain_turn_count
        scores["overall"] = {'em': round(em_total / max(1, turn_count) * 100, 2),
                            'f1': round(f1_total / max(1, turn_count) * 100, 2),
                            'turns': turn_count}

        return scores

    def compute(self, pred_data, ref_data):
        gold_data, _ = self.gold_answers_to_dict_from_data(ref_data)

        exact_scores, f1_scores = self.get_raw_scores(pred_data, gold_data)
        em_total, f1_total, turn_count = 0, 0, 0
        for key in exact_scores:
            assert key in f1_scores
            em_total += exact_scores.get(key, 0)
            f1_total += f1_scores.get(key, 0)
            turn_count += 1

        em = round(em_total / max(1, turn_count) * 100, 2)
        f1 = round(f1_total / max(1, turn_count) * 100, 2)
        total = turn_count
        return {"exact_match": em, "f1": f1, "total": total}
    
    def compute_all(self, pred_data, gold_file, output_file):
        gold_data, id_to_source = self.gold_answers_to_dict_from_file(gold_file)

        human_scores = json.dumps(self.human_performance(gold_data, id_to_source), indent=2)
        model_scores = json.dumps(self.model_performance(pred_data, gold_data, id_to_source), indent=2)
        print("human_scores", human_scores)
        print("model_scores", model_scores)
        with open(output_file, "w") as f:
            f.write("human scores\n")
            f.write(human_scores+"\n")
            f.write("model scores\n")
            f.write(model_scores+"\n")


class QuacEvaluator(Evaluator):
    def __init__(self):
        pass
    
    def gold_answers_to_dict_from_data(self, ref_data):
        gold_dict = {}
        for item in ref_data:
            story_id, turn_id = self._parse_pred_key(item["id"])
            gold_answers = item["answers"]["text"]

            key = (story_id, turn_id)
            gold_dict[key] = gold_answers
        return gold_dict

    def _parse_pred_key(self, pred_key):
        splits = pred_key.split("_q#")
        assert len(splits) == 2
        turn_id = splits[-1]
        story_id = splits[0]
        return str(story_id), int(turn_id)
    
    def _get_pred_key(self, story_id, turn_id):
        pred_key = story_id + "_q#" + str(turn_id)
        return pred_key
    
    @staticmethod
    def is_overlapping(x1, x2, y1, y2):
        return max(x1, y1) <= min(x2, y2)
    
    @staticmethod
    def display_counter(title, c, c2=None):
        print(title)
        for key, _ in c.most_common():
            if c2:
                print('%s: %d / %d, %.1f%%, F1: %.1f' % (
                key, c[key], sum(c.values()), c[key] * 100. / sum(c.values()), sum(c2[key]) * 100. / len(c2[key])))
            else:
                print('%s: %d / %d, %.1f%%' % (key, c[key], sum(c.values()), c[key] * 100. / sum(c.values())))
    
    @staticmethod    
    def compute_span_overlap(pred_span, gt_span, context=None):
        if gt_span == 'CANNOTANSWER':
            if pred_span == 'CANNOTANSWER':
                return 'Exact match', 1.0
            return 'No overlap', 0.
        fscore = Evaluator.compute_f1(pred_span, gt_span)
        if not context:
           overlap = False
        else: 
            pred_start = context.find(pred_span)
            gt_start = context.find(gt_span)

            if pred_start == -1 or gt_start == -1:
                return 'Span indexing error', fscore

            pred_end = pred_start + len(pred_span)
            gt_end = gt_start + len(gt_span)

            overlap = QuacEvaluator.is_overlapping(pred_start, pred_end, gt_start, gt_end)

        fscore = QuacEvaluator.compute_f1(pred_span, gt_span)

        if Evaluator.compute_exact(pred_span, gt_span):
            return 'Exact match', fscore
        if overlap:
            return 'Partial overlap', fscore
        else:
            return 'No overlap', fscore


    @staticmethod
    def metric_max_over_ground_truths(prediction, ground_truths, context=None):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = QuacEvaluator.compute_span_overlap(prediction, ground_truth, context=context)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths, key=lambda x: x[1])


    @staticmethod
    def leave_one_out_max(prediction, ground_truths, context=None):
        if len(ground_truths) == 1:
            scores = QuacEvaluator.metric_max_over_ground_truths(prediction, ground_truths, context=context)
            return int(scores[0] == 'Exact match'), scores[1]
        else:
            t_em = []
            t_f1 = []
            # leave out one ref every time
            for i in range(len(ground_truths)):
                idxes = list(range(len(ground_truths)))
                idxes.pop(i)
                refs = [ground_truths[z] for z in idxes]
                scores = QuacEvaluator.metric_max_over_ground_truths(prediction, refs, context=context)
                t_em.append(int(scores[0] == 'Exact match'))
                t_f1.append(scores[1])
        return 1.0 * sum(t_em) / len(t_em), 1.0 * sum(t_f1) / len(t_f1)

    @staticmethod
    def handle_cannot(refs):
        num_cannot = 0
        num_spans = 0
        for ref in refs:
            if ref == 'CANNOTANSWER':
                num_cannot += 1
            else:
                num_spans += 1
        if num_cannot >= num_spans:
            refs = ['CANNOTANSWER']
        else:
            refs = [x for x in refs if x != 'CANNOTANSWER']
        return refs

    @staticmethod
    def leave_one_out(refs):
        if len(refs) == 1:
            return 1.
        splits = []
        for r in refs:
            splits.append(r.split())
        t_f1 = 0.0
        for i in range(len(refs)):
            m_f1 = 0
            for j in range(len(refs)):
                if i == j:
                    continue
                f1_ij = Evaluator.compute_f1(refs[i], refs[j])
                if f1_ij > m_f1:
                    m_f1 = f1_ij
            t_f1 += m_f1
        return t_f1 / len(refs)

    def eval_fn(self, val_results, model_results, verbose=False):
        min_f1 = 0.4
        span_overlap_stats = Counter()
        sentence_overlap = 0.
        para_overlap = 0.
        total_qs = 0.
        f1_stats = defaultdict(list)
        unfiltered_ems = []
        unfiltered_f1s = []
        human_f1 = []
        HEQ = 0.
        DHEQ = 0.
        total_dials = 0.
        # yes_nos = []
        # followups = []
        unanswerables = []
        for p in val_results:
            for par in p['paragraphs']:
                did = par['id']
                qa_list = par['qas']
                good_dial = 1.
                for qa in qa_list:
                    q_idx = qa['id']
                    val_spans = [anss['text'] for anss in qa['answers']]
                    val_spans = QuacEvaluator.handle_cannot(val_spans)
                    hf1 = QuacEvaluator.leave_one_out(val_spans)

                    if did not in model_results or q_idx not in model_results[did]:
                        print(did, q_idx, 'no prediction for this dialogue id')
                        good_dial = 0
                        f1_stats['NO ANSWER'].append(0.0)
                        # yes_nos.append(False)
                        # followups.append(False)
                        if val_spans == ['CANNOTANSWER']:
                            unanswerables.append(0.0)
                        total_qs += 1
                        unfiltered_ems.append(0.0)
                        unfiltered_f1s.append(0.0)
                        if hf1 >= min_f1:
                            human_f1.append(hf1)
                        continue

                    # pred_span, pred_yesno, pred_followup = model_results[did][q_idx]
                    pred_span = model_results[did][q_idx]

                    max_overlap, _ = QuacEvaluator.metric_max_over_ground_truths( \
                        pred_span, val_spans, par['context'])
                    max_em, max_f1 = QuacEvaluator.leave_one_out_max( \
                        pred_span, val_spans, par['context'])
                    unfiltered_ems.append(max_em)
                    unfiltered_f1s.append(max_f1)

                    # dont eval on low agreement instances
                    if hf1 < min_f1:
                        continue

                    human_f1.append(hf1)
                    # yes_nos.append(pred_yesno == qa['yesno'])
                    # followups.append(pred_followup == qa['followup'])
                    if val_spans == ['CANNOTANSWER']:
                        unanswerables.append(max_f1)
                    # if verbose:
                    #     print("-" * 20)
                    #     print(pred_span)
                    #     print(val_spans)
                    #     print(max_f1)
                    #     print("-" * 20)
                    if max_f1 >= hf1:
                        HEQ += 1.
                    else:
                        good_dial = 0.
                    span_overlap_stats[max_overlap] += 1
                    f1_stats[max_overlap].append(max_f1)
                    total_qs += 1.
                DHEQ += good_dial
                total_dials += 1
        DHEQ_score = 100.0 * DHEQ / total_dials
        HEQ_score = 100.0 * HEQ / total_qs
        all_f1s = sum(f1_stats.values(), [])
        overall_f1 = 100.0 * sum(all_f1s) / len(all_f1s)
        unfiltered_f1 = 100.0 * sum(unfiltered_f1s) / len(unfiltered_f1s)
        # yesno_score = (100.0 * sum(yes_nos) / len(yes_nos))
        # followup_score = (100.0 * sum(followups) / len(followups))
        yesno_score = 0.0
        followup_score = 0.0
        unanswerable_score = (100.0 * sum(unanswerables) / len(unanswerables))
        metric_json = {"unfiltered_f1": unfiltered_f1, "f1": overall_f1, "HEQ": HEQ_score, "DHEQ": DHEQ_score, "yes/no": yesno_score, "followup": followup_score, "unanswerable_acc": unanswerable_score}
        if verbose:
            print("=======================")
            QuacEvaluator.display_counter('Overlap Stats', span_overlap_stats, f1_stats)
            print("=======================")
            print('Overall F1: %.1f' % overall_f1)
            # print('Yes/No Accuracy : %.1f' % yesno_score)
            # print('Followup Accuracy : %.1f' % followup_score)
            print('Unfiltered F1 ({0:d} questions): {1:.1f}'.format(len(unfiltered_f1s), unfiltered_f1))
            print('Accuracy On Unanswerable Questions: {0:.1f} %% ({1:d} questions)'.format(unanswerable_score, len(unanswerables)))
            print('Human F1: %.1f' % (100.0 * sum(human_f1) / len(human_f1)))
            print('HEQQ | Model F1 >= Human F1 (Questions): %d / %d, %.1f%%' % (HEQ, total_qs, 100.0 * HEQ / total_qs))
            print('HEQD | Model F1 >= Human F1 (Dialogs): %d / %d, %.1f%%' % (DHEQ, total_dials, 100.0 * DHEQ / total_dials))
            print("=======================")
        return metric_json          
    
    def compute(self, pred_data, ref_data):
        gold_data = self.gold_answers_to_dict_from_data(ref_data)

        min_F1, total = 0.4, 0
        all_em, all_f1, hf1 = [], [], []

        for idx, row in enumerate(pred_data):
            key, pred = row["id"], row["prediction_text"]
            story_id, turn_id = self._parse_pred_key(key)
            gold_key = (story_id, turn_id)
            if gold_key not in gold_data:
                sys.stderr.write(f'Missing reference for {key}\n')
            
            clean_t = QuacEvaluator.handle_cannot(gold_data[(story_id, turn_id)])
            # compute human performance
            human_F1 = QuacEvaluator.leave_one_out(clean_t)
            if human_F1 < min_F1: 
                continue
            
            total += 1
            hf1.append(human_F1)
            scores = QuacEvaluator.leave_one_out_max(pred, clean_t)
            all_em.append(scores[0])
            all_f1.append(scores[1])

        em = 100.0 * sum(all_em) / total
        f1 = 100.0 * sum(all_f1) / total
        human_f1 = 100.0 * sum(hf1) / total
        return {"human_f1": human_f1, "exact_match": em, "f1": f1, "total": total}
    
    def compute_all(self, pred_data, gold_file, output_file):
        val = json.load(open(gold_file, 'r'))['data']
        preds = defaultdict(dict)
        total = 0
        val_total = 0
        for row in pred_data:
            qid = row["id"]
            dia_id, _ = self._parse_pred_key(qid)
            preds[dia_id][qid] = row["prediction_text"]
            total += 1

        for p in val:
            for par in p['paragraphs']:
                did = par['id']
                qa_list = par['qas']
                val_total += len(qa_list)
        
        model_scores = json.dumps(self.eval_fn(val, preds, verbose=True), indent=2)
        with open(output_file, "w") as f:
            f.write(model_scores+"\n")



ConvqaEvaluators = {
    "coqa": CoQAEvaluator,
    "quac": QuacEvaluator,
}


def main(args):
    if args.data == "coqa":
        evaluator = CoQAEvaluator()

        with open(args.pred_file) as f:
            pred_data = Evaluator.preds_to_rows(args.pred_file)
        print(json.dumps(evaluator.compute_all(pred_data, args.data_file, args.out_file), indent=2))
    else:
        evaluator = QuacEvaluator()

        with open(args.pred_file) as f:
            pred_data = Evaluator.preds_to_rows(args.pred_file)
        print(json.dumps(evaluator.compute_all(pred_data, args.data_file, args.out_file), indent=2))


# python src/modules/convqa_evaluator.py --data coqa --pred-file save/roberta-base-coqa/predictions_test.json --data-file data/coqa/coqa-dev-v1.0.json --out-file save/roberta-base-coqa/all_test_results.txt
# python src/modules/convqa_evaluator.py --data quac --pred-file save/roberta-base-quac/predictions_test.json --data-file data/quac/val_v0.2.json --out-file save/roberta-base-quac/all_test_results.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Official evaluation script for CoQA.')
    parser.add_argument('--data', dest="data", help='Input data JSON file.')
    parser.add_argument('--data-file', dest="data_file", help='Input data JSON file.')
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.')
    parser.add_argument('--out-file', '-o', metavar='save',
                        help='The folder to save the evaluation results.')
    parser.add_argument('--verbose', '-v', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    main(args)