import os
import argparse
import openai
import pickle
import spacy
import numpy as np
import random

from tqdm import tqdm

from utils import *
from few_shot import convert_paragraphs_to_context
from dataset_utils import read_hotpot_data, hotpot_evaluation, hotpot_evaluation_with_multi_answers, f1auc_score, read_incorrect_answers, normalize_answer
from comp_utils import safe_completion, length_of_prompt, conditional_strip_prompt_prefix, get_similarity


TEST_PART = 250

_MAX_TOKENS = 144

nlp = spacy.load('en_core_web_sm')

# PROMOT CONTROL
PE_STYLE_SEP = " The reason is as follows."
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)

    parser.add_argument('--style', type=str, default="p-e")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=30)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=308)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result',  default=False, action='store_true')

    args = parser.parse_args()
    specify_engine(args)
    return args


def extract_stem_tokens(text):
    doc = nlp(text)
    stem_tokens = []
    for i, t in enumerate(doc):
        pos, tag = t.pos_, t.tag_
        if pos == 'AUX':
            continue
        is_stem = False
        if tag.startswith('NN'):
            is_stem = True
        if tag.startswith('VB'):
            is_stem = True
        if tag.startswith('JJ'):
            is_stem = True
        if tag.startswith('RB'):
            is_stem = True
        if tag == 'CD':
            is_stem = True
        if is_stem:
            stem_tokens.append({
                'index': i,
                'text': t.text,
                'lemma': t.lemma_,
                'pos': t.pos_,
                'tag': t.tag_
            })
    return stem_tokens


def rationale_coverage_quality(r, q):
    q_stem_tokens = extract_stem_tokens(q)
    r_stem_tokens = extract_stem_tokens(r)
    r_lemma_tokens = [x['lemma'] for x in r_stem_tokens]
    q_lemma_tokens = [x['lemma'] for x in q_stem_tokens]

    hit = 0
    for t in r_lemma_tokens:
        if t in q_lemma_tokens:
            hit += 1

    return hit / len(r_lemma_tokens)

def result_cache_name(args):
    return "misc/manual{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.dat".format(args.annotation, args.engine_name,
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.dev_slice + args.num_dev,
        args.num_distractor, args.style)

# return prompt stop_signal
def prompt_for_manual_prediction(ex, shots, style):
    stop_signal = "\n\n"
    # P-E
    if style == "p-e":
        showcase_examples = [
            "{}\nQ: {}\nA: {}.{} {}\n".format(
                convert_paragraphs_to_context(s), s["question"],
                s["answer"], PE_STYLE_SEP, s["manual_rationale"]) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(convert_paragraphs_to_context(ex), ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    # E-P
    elif style == "e-p":
        showcase_examples = [
            "{}\nQ: {}\nA: {}{} {}.\n".format(
                convert_paragraphs_to_context(s), s["question"], s["manual_rationale"], EP_STYLE_SEP,
                s["answer"]) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(convert_paragraphs_to_context(ex), ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")
    return prompt, stop_signal

def in_context_manual_prediction(ex, training_data, engine, style="p-e", length_test_only=False):
    prompt, stop_signal = prompt_for_manual_prediction(ex, training_data, style)
    #print("prompt:",prompt)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_TOKENS)
        print(prompt)
        return pred
    else:
        pred = safe_completion(engine, prompt, _MAX_TOKENS, stop_signal, temp=0.0, logprobs=5)


    #print("pred:",pred)
    # pred["id"] = ex["id"]
    # pred["prompt"] = prompt
    # if len(pred["text"]) > len(prompt):
    #     pred["text"] = pred["text"][len(prompt):]
    # else:
    #     pred["text"] = "null"
    # pred["completion_offset"] = len(prompt)
    #print("text:",pred["text"])

    return pred

def get_sep_text(pred, style):
    if style == "e-p":
        for sep in EP_POSSIBLE_SEP_LIST:
            if sep in pred:
                return sep
        return None
    else:
        raise RuntimeError("Unsupported decoding style")

def post_process_manual_prediction(p,newdict,count,style):
    text = p[0]
    text = text.strip()

    # place holder
    answer = "null"
    rationale = "null"
    rationale_indices = []
    if style == "p-e":
        sep = PE_STYLE_SEP
        if sep in text:
            segments = text.split(sep)
            answer = segments[0].strip().strip('.')
            rationale = segments[1].strip()
    elif style == "e-p":
        sep = get_sep_text(p[0], style)
        if sep is not None:
            segments = text.split(sep)
            answer = segments[1].strip().strip('.')
            rationale = segments[0].strip()
        else:
            answer = text
    else:
        raise RuntimeError("Unsupported decoding style")

    newdict[str(count)+"answer"] = answer
    print("answer:",answer)
    newdict[str(count)+"rationale"] = rationale
    print("rationale:",rationale)
    newdict[str(count)+"rationale_indices"] = rationale_indices
    return answer, rationale

def post_process_manual_confidance(pred, style):
    completion_offset = pred["completion_offset"]
    tokens = pred["logprobs"]["tokens"]
    token_offset = pred["logprobs"]["text_offset"]

    completion_start_tok_idx = token_offset.index(completion_offset)
    # exclusive idxs
    if "<|endoftext|>" in tokens:
        completion_end_tok_idx = tokens.index("<|endoftext|>") + 1
    else:
        completion_end_tok_idx = len(tokens)
    completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
    completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]

    if style == "p-e":
        if PE_STYLE_SEP in pred["text"]:
            sep_token_offset = completion_offset + pred["text"].index(PE_STYLE_SEP)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            ans_logprob = sum(completion_probs[:sep_start_idx - 1])
            rat_logprob = sum(completion_probs[(sep_start_idx + 6):])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0
    elif style == "e-p":
        sep_text = get_sep_text(pred, style)
        if sep_text is not None:
            sep_token_offset = completion_offset + pred["text"].index(sep_text)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            rat_logprob = sum(completion_probs[:sep_start_idx + 3])
            ans_logprob = sum(completion_probs[(sep_start_idx + 3):-1])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0
    else:
        raise RuntimeError("Unsupported decoding style")

    pred["answer_logprob"] = ans_logprob
    pred["rationale_logprob"] = rat_logprob
    pred["joint_lobprob"] = ans_logprob + rat_logprob
    return ans_logprob, rat_logprob

def post_process_manual_prediction_and_confidence(pred,newdict,count,style):
    # process answer and rationale
    post_process_manual_prediction(pred,newdict,count,style)
    #post_process_manual_confidance(pred, style)


def evaluate_manual_predictions(dev_set, newdict, style="p-e", do_print=False):
    acc_records = []
    rat_records = []
    f1_records, pre_records, rec_records = [], [], []
    logprob_records = []
    ansprob_records = []

    certified_incorrect_answers = read_incorrect_answers()
    count=0
    score1=0
    score2=0
    for ex in dev_set:
        gt_rat = ' '.join(ex['rationale'])
        p_ans = newdict[str(count)+'answer']
        p_rat = newdict[str(count)+'rationale']
        count+=1
        print("answer:",p_ans)
        print("ground_thruth:",ex["answer_choices"])
        score1+=rationale_coverage_quality(p_rat,convert_paragraphs_to_context(ex))
        score2+=get_similarity(p_rat,convert_paragraphs_to_context(ex))
        acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
        acc_records.append(acc)
    mean_of_array = lambda x: sum(x) / len(x)
    print("accuracy:", mean_of_array(acc_records))
    print("lexical similarity between generated test example explanation and test example premise:",score1/20)
    print("cosine similarity between generated test example explanation and test example premise:",score2/20)



    # for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):

    #     gt_rat = ' '.join(ex['rationale'])
    #     p_ans = pred['answer']
    #     p_rat = pred['rationale']
    #     acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
    #     acc_records.append(acc)
    #     rat_acc = False
    #     rat_records.append(rat_acc)
    #     f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
    #     logprob_records.append(pred['joint_lobprob'])
    #     ansprob_records.append(pred['answer_logprob'])
    #     if do_print and not acc:
    #         if ex['id'] in certified_incorrect_answers and p_ans in certified_incorrect_answers[ex['id']]:
    #             continue
    #         print("--------------{} EX {} RAT {} F1 {:.2f}--------------".format(idx, acc, rat_acc, f1))
    #         print(convert_paragraphs_to_context(ex))
    #         print(ex['question'])

    #         print('\nRAW TEXT', '[' + pred['text'].strip() + ']')
    #         print('PR ANS:', p_ans)
    #         # print('PR RAT:', p_rat)
    #         print('GT ANS:', gt_ans)
    #         print(json.dumps({'qas_id': ex['id'], 'answer': p_ans}))

    # mean_of_array = lambda x: sum(x) / len(x)
    # print("EX", mean_of_array(acc_records), "RAT", mean_of_array(rat_records))
    # print("F1: {:.2f}".format(mean_of_array(f1_records)),
    #         "PR: {:.2f}".format(mean_of_array(pre_records)),
    #         "RE: {:.2f}".format(mean_of_array(rec_records)))
    # print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
    #         ansprob_records, acc_records)))

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_hotpot_data(f"data/sim_train.json", args.num_distractor, manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    # file = open("misc/manualstd_sim_text-davinci-001_tr0-30_dv0-308_nds2_e-p_predictions.dat",'rb')
    # newdict1 = pickle.load(file)
    # file.close()

    random.seed(7)

    rand_dev_set = random.sample(dev_set,20)

    pairwise_ice_sim = []

    # for i, x in tqdm(enumerate(train_set), total=len(train_set), desc="Predicting"):
    #     sim = []
    #     for y in train_set:
    #         sim.append(get_similarity(convert_paragraphs_to_context(x),convert_paragraphs_to_context(y)))                
    #     pairwise_ice_sim.append(sim)

    # pairwise_ice_sim = np.array(pairwise_ice_sim)

    # with open('pairwise_ice_bcs_sim.pkl', 'wb') as f:
    #     pickle.dump(pairwise_ice_sim, f)

    # exit(0)
    file = open("pairwise_ice_bcs_sim.pkl",'rb')
    pairwise_ice_sim = pickle.load(file)
    file.close()

    predictions=[]

    averages = []

    for x in tqdm(rand_dev_set, total=len(rand_dev_set), desc="Predicting"):

        similarities = []
        for ex in train_set:
            similarities.append(get_similarity(convert_paragraphs_to_context(x),convert_paragraphs_to_context(ex)))
        
        similarities = np.array(similarities).reshape(-1, 1)
        
        beta=0.75
        selected_data_indices = []
        unselected_data_indices = list(range(len(train_set)))

        # find the most similar phrase (using original cosine similarities)
        best_idx = np.argmax(similarities)
        selected_data_indices.append(best_idx)
        unselected_data_indices.remove(best_idx)

        for _ in range(4):
            unselected_data_distances_to_text = similarities[unselected_data_indices, :]
            unselected_data_distances_pairwise = pairwise_ice_sim[unselected_data_indices][:,selected_data_indices]

            # find new candidate with
            idx = int(np.argmax(
                beta * unselected_data_distances_to_text - (1 - beta) * np.max(unselected_data_distances_pairwise,
                                                                                 axis=1).reshape(-1, 1)))
            best_idx = unselected_data_indices[idx]

            # select new best phrase and update selected/unselected phrase indices list
            selected_data_indices.append(best_idx)
            unselected_data_indices.remove(best_idx)
        
        new_train_set = []
        for i in selected_data_indices:
            new_train_set.append(train_set[i])
        
        averages.append(sum(similarities[selected_data_indices])/len(selected_data_indices))
        
        predictions.append(in_context_manual_prediction(x, new_train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test))


    if args.run_length_test:
        print(result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_TOKENS)
        return
    # save
    newdict={}
    count=0
    for p in predictions:
        post_process_manual_prediction_and_confidence(p,newdict,count,args.style)
        count+=1

    filename=result_cache_name(args)
    file = open(filename,'wb')    #data we wrote in file
    pickle.dump(newdict,file)
    file.close()
        # read un indexed dev
    #print(predictions)
    #     dump_json(predictions, result_cache_name(args))
    # [post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]
    # # acc
    analyze_few_shot_manual_prediction(rand_dev_set,args)

    print("Average Similarity: ", sum(averages)/len(averages))

def analyze_few_shot_manual_prediction(rand_dev_set,args):
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    #predictions = read_json(result_cache_name(args))
    #[post_process_manual_prediction_and_confidence(p, args.style) for p in predictions]
    file = open(result_cache_name(args),'rb')
    newdict = pickle.load(file)
    file.close()

    if args.show_result:
        dev_set = dev_set[-TEST_PART:]
        #predictions = predictions[-TEST_PART:]

    evaluate_manual_predictions(rand_dev_set, newdict, args.style, do_print=False)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_manual_prediction(args)
    else:
        analyze_few_shot_manual_prediction(args)