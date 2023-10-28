import argparse
import pickle
import spacy
import random
import pandas as pd

from tqdm import tqdm

from utils import *
from few_shot import convert_paragraphs_to_context
from dataset_utils import read_hotpot_data, hotpot_evaluation, hotpot_evaluation_with_multi_answers, f1auc_score, read_incorrect_answers, normalize_answer
from comp_utils import safe_completion, length_of_prompt, conditional_strip_prompt_prefix, get_similarity

from joblib import Parallel, delayed

TEST_PART = 250

_MAX_TOKENS = 145

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
    parser.add_argument('--num_shot', type=int, default=32)
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
    return "misc/manual{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_highest_self_consistent.dat".format(args.annotation, args.engine_name,
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
        # print(prompt)
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
    # print("answer:",answer)
    newdict[str(count)+"rationale"] = rationale
    # print("rationale:",rationale)
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
    for ex in dev_set:
        # if(count==20):
        #     break
        gt_rat = ' '.join(ex['rationale'])
        p_ans = newdict[str(count)+'answer']
        p_rat = newdict[str(count)+'rationale']
        count+=1
        #print("answer:",p_ans)
        #print("ground_thruth:",ex["answer_choices"])
        score1+=get_similarity(p_rat,convert_paragraphs_to_context(ex))
        acc, (f1, pre, rec), gt_ans = hotpot_evaluation_with_multi_answers(p_ans, ex["answer_choices"])
        acc_records.append(acc)
    mean_of_array = lambda x: sum(x) / len(x)
    n=len(acc_records)
    print("accuracy:", mean_of_array(acc_records))
    print("cosine similarity between generated explanation and test example context:",score1/n)

    with open('acc_highest_self_consistency.pkl', 'wb') as f:
        pickle.dump(acc_records, f)

    #print("similarity between generated test example explanation and test example premise:",score1/250)
    


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

    print(len(dev_set), len(train_set))

    #Random seeding
    random.seed(100)
    
    train_set_indices=[]
    for i in range(0,len(train_set)):
        train_set_indices.append(i)
    

    matrix_ic_test = [[0 for i in range(len(train_set))] for j in range(len(dev_set))]
    predictions = {} 
    # overall_incontext_list = []


    def self_consistency_based_selection(ex,number):

        expls_per_test=[]
        incontext_ex = []

        # Generating 10 sets of 5 ic examples each
        for _ in range(0, 10):
            #Each set of size 5
            rand_indices = random.sample(train_set_indices, 5)
            incontext_ex.append(rand_indices)

            train_set_new=[]
            for indices in rand_indices:
                train_set_new.append(train_set[indices])

            genexp = in_context_manual_prediction(ex, train_set_new, engine=args.engine, style=args.style, length_test_only=args.run_length_test)

            expls_per_test.append(genexp)
        
        print("\n\nPrinting all explanations for test example",number,"\n")
        print(expls_per_test,"\n\n")
        #For all test ex 
        # overall_incontext_list.append(incontext_ex)

        #Self-consistency on 10 gen_exp
        matches = [0]*len(expls_per_test)
        matches_list = []
        avg_match_score = [0]*len(expls_per_test)
        cur_index = 0
        for i in range(len(expls_per_test)):
            overlap_score_each = []
            temp_list = []
            for j in range(len(expls_per_test)):
                if(i!=j):
                    #overlap_score=rationale_coverage_quality(expls_per_test[i][0], expls_per_test[j][0])
                    overlap_score=get_similarity(expls_per_test[i][0], expls_per_test[j][0])
                    overlap_score_each.append(overlap_score)

            #print("overlap scores")
            #print(overlap_score_each)

            for j in range(len(expls_per_test)-1):           
                # Threshold of some% match fixed
                if(overlap_score_each[j]>=0.97):
                    matches[cur_index]+=1
                    if(len(temp_list)==0): avg_match_score[cur_index] = overlap_score_each[j]
                    else: avg_match_score[cur_index] = (avg_match_score[cur_index]*len(temp_list) + overlap_score_each[j])/(len(temp_list)+1)
                    if(j>=i):
                        temp_list.append(j+1)
                    else:
                        temp_list.append(j)
            temp_list.append(i)
            matches_list.append(temp_list)
            print("For gen_expl", cur_index, "the overlap scores with other explanations for test ex",number)
            print(overlap_score_each)
            print("Matches for gen_expl", cur_index, "are", matches)
            print("Match List for gen_expl", cur_index, "are", matches_list)
            print("*************************************************************")
            cur_index+=1

        max_index = -1
        max_val = -1
        #Here we are finding the highest match as well as highest avg overlap one (eg: if 3 expln are of same match then we will resolve on the basis of avg overlap)
        for i in range(0, len(expls_per_test)):
            if(matches[i]>max_val):
                max_val=matches[i]
                max_index = i
            elif(matches[i]==max_val):
                if(avg_match_score[i]>avg_match_score[max_index]):
                    max_index = i
        print("Match count for test example", number)
        print(matches)
        print("Avg overlap score")
        print(avg_match_score)
        print("Highest match index = ", max_index)
        #Here we will get the expln indices from which the highest match one is matching with
        print("Matched expln indices")
        print(matches_list[max_index])

        #Matrix val = 1 for ice of highest matching expln index
        for ind in incontext_ex[max_index]:
            matrix_ic_test[number][ind] = 1

        # #Matrix val = 1 for all ice of all the matching expln indices
        # for max_ind in matches_list[max_index]:
        #     for ind in incontext_ex[max_ind]:
        #         matrix_ic_test[number][ind] = 1

        print("Test example // selected in-context examples")
        print(number ," // ", matrix_ic_test[number])

        ICE_set=[]
        for ind in incontext_ex[max_index]:
            ICE_set.append(train_set[ind])
        prompt, stop_signal = prompt_for_manual_prediction(ex, ICE_set, style=args.style)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("FINAL Selected in-context examples for test example",number,"\n")
        print(prompt)

        print("Generated explanation for test example ",number,"using selected ICE is")
        print(expls_per_test[max_index])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        #It is for storing the best explanation
        predictions[number]=expls_per_test[max_index]

        df = pd.DataFrame({'matrix_ic_test': matrix_ic_test}) 
        df_filename = "matrix_ice_highest_consistent.csv"
        df.to_csv(df_filename, index=False)

        # with open("matrix_2.pkl", 'wb') as f:
        #     pickle.dump(matrix_ic_test, f)


    ############################################################################################
    Parallel(n_jobs=5, backend="threading")(delayed(self_consistency_based_selection)(ex,ind) for ind, ex in enumerate(dev_set))    
    ############################################################################################


    print("Pickling...")
    newdict={}
    count=0
    for key,p in predictions.items():
        post_process_manual_prediction_and_confidence(p,newdict,count,args.style)
        count+=1

    filename=result_cache_name(args)
    file = open(filename,'wb')    #data we wrote in file
    pickle.dump(newdict,file)
    file.close()
    print("Complete")

    analyze_few_shot_manual_prediction(args)

def analyze_few_shot_manual_prediction(args):
    dev_set = read_hotpot_data(f"data/sim_dev.json", args.num_distractor)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    file = open(result_cache_name(args),'rb')
    newdict = pickle.load(file)
    file.close()

    # if args.show_result:
    #     dev_set = dev_set[-TEST_PART:]
        #predictions = predictions[-TEST_PART:]

    evaluate_manual_predictions(dev_set, newdict, args.style, do_print=False)
    print(result_cache_name(args))


if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_manual_prediction(args)
    else:
        analyze_few_shot_manual_prediction(args)

