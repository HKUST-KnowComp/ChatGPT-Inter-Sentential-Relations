import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai==0.28'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'backoff'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'retry'])

import os
import openai
import time
import backoff
import re
import json
from retry import retry
from tqdm import tqdm


# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@retry()
def get_response(model, prompt, temperature, max_tokens, top_p):
    response = completions_with_backoff(model=model, messages=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
#     time.sleep(3)
    return response


def read_data(filename):
    dataset = []
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        dataset.append(result)
    return dataset


def pre_processing_data_PDTB(dataset):
    processed_data = []
    incontext_prompt="All answer select from following:\nA. Comparison.Concession, nonetheless \nB. Comparison.Contrast, however\nC. Contingency.Cause, so\nD. Contingency.Pragmatic Cause, indeed\nE. Expansion.Alternative, instead\nF. Expansion.Conjunction, also\nG. Expansion.Instantiation, for example\nH. Expansion.List, and\nI. Expansion.Restatement, specifically\nJ. Temporal.Asynchronous, before\nK. Temporal.Synchrony, when\n\nArgument 1: \"Coke could be interested in more quickly developing some of the untapped potential in those markets.\"\nArgument 2: \"A Coke spokesman said he couldn't say whether that is the direction of the talks.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Comparison.Concession, nonetheless\n\nArgument 1: \"Tanks currently are defined as armored vehicles weighing 25 tons or more that carry large guns.\"\nArgument 2: \"The Soviets complicated the issue by offering to include light tanks, which are as light as 10 tons.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Comparison.Contrast, however\n\nArgument 1: \"The male part, the anthers of the plant, and the female, the pistils, of the same plant are within a fraction of an inch or even attached to each other.\"\nArgument 2: \"The anthers in these plants are difficult to clip off.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Contingency.Cause, so\n\nArgument 1: \"I'm not accusing insurers of dereliction of duty, Robert Patricelli of the U.S. Chamber of Commerce told Mr. Waxman's panel.\"\nArgument 2: \"You can't ask one carrier to underwrite on social grounds when that might destroy it in the marketplace.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Contingency.Pragmatic Cause, indeed\n\nArgument 1: \"The judge, suspended from his bench pending his trial, which began this week, vehemently denies all the allegations against him, calling them \"ludicrous\" and \"imaginative, political demagoguery.\" \nArgument 2: \"He blames the indictment on local political feuding, unhappiness with his aggressive efforts to clear the courthouse's docket and a vendetta by state investigators and prosecutors angered by some of his rulings against them.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Alternative, instead\n\nArgument 1: \"Baskets of roses and potted palms adorned his bench.\"\nArgument 2: \"The local American Legion color guard led the way.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Conjunction, also\n\nArgument 1: \"In recent years, demand for hybrid seeds has spurred research at a number of chemical and biotechnology companies, including Monsanto Co., Shell Oil Co. and Eli Lilly & Co.\"\nArgument 2: \"One technique developed by some of these companies involves a chemical spray supposed to kill only a plant's pollen.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Instantiation, for example\n\nArgument 1: \"Accepted bids ranged from 8% to 8.019%.\"\nArgument 2: \"The bank holding company will auction another $50 million in each maturity next Tuesday.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.List, and\n\nArgument 1: \"His recent appearance at the Metropolitan Museum, dubbed \"A Musical Odyssey,\" was a case in point.\"\nArgument 2: \"It felt more like a party, or a highly polished jam session with a few friends, than a classical concert.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Restatement, specifically\n\nArgument 1: \"For example, the chief executive himself now pays 20% of the cost of his health benefits;\"\nArgument 2: \"the company used to pay 100%.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Temporal.Asynchronous, before\n\nArgument 1: \"Panamanian dictator Torrijos, he was told, had granted the shah of Iran asylum in Panama as a favor to Washington.\"\nArgument 2: \"Mr. Sanford was told Mr. Noriega's friend, Mr. Wittgreen, would be handling the shah's security.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Temporal.Synchrony, when"
    for i in range(len(dataset)):
        input_text= "Argument 1: \"" + dataset[i]['Arg1RawText'] + ".\"\nArgument 2: \"" + dataset[i]['Arg2RawText'] + ".\"\nQuestion: What is the discourse relation and connective between Argument 1 and Argument 2? \nA. Comparison.Concession, if \nB. Comparison.Contrast, however \nC. Contingency.Cause, so \nD. Contingency.Pragmatic, indeed \nE. Expansion.Alternative, instead \nF. Expansion.Conjunction, also \nG. Expansion.Instantiation, for example \nH. Expansion.List, and \nI. Expansion.Restatement, specifically \nJ. Temporal.Asynchronous, before \nK. Temporal.Synchrony, when \nAnswer:"
        prompt_input = [{"role": "user", "content": incontext_prompt+input_text}]
        processed_data.append(prompt_input)
    return processed_data


def get_knowledge(prompt_input, model, temperature, max_tokens, top_p):
    ChatGPT_answer_text = []
    save_knowledge_filename = 'PDTBJI-IMP_Test_ChatGPT_Top_Second_Connective_InContextLearning_v1106.json' 
    
    for i in tqdm(range(len(prompt_input))):
        response = get_response(model, prompt_input[i], temperature, max_tokens, top_p)
        if i % 50 == 0:
            print(prompt_input[i])
            print(response)
            save_json(save_knowledge_filename, ChatGPT_answer_text)
        ChatGPT_answer_text.append(response)
    return ChatGPT_answer_text

def save_json(save_file_name, response):
    with open(save_file_name, 'w') as json_file:
        json.dump(response, json_file)

if __name__== "__main__":
    openai.api_key = " " 
#     file_name = "./DiscoPrompt/PDTB2-Exp/test.jsonl"
    file_name = "./DiscoPrompt/PDTB2-Imp/test.jsonl" 
    dataset = read_data(file_name)
    prompt_input= pre_processing_data_PDTB(dataset)
    
    model="gpt-3.5-turbo-1106"
    temperature=1
    top_p = 1
    max_tokens=256
    ChatGPT_answer_text = get_knowledge(prompt_input, model, temperature, max_tokens, top_p)

    save_knowledge_filename = 'PDTBJI-IMP_Test_ChatGPT_Top_Second_Connective_InContextLearning_All_v1106.json' #'PDTBJI_dev_list_knowledge_2048tokens.json'
    save_json(save_knowledge_filename, ChatGPT_answer_text)


## ChatGPT_Prompt Template for discourse relation (PDTB Top level)
#         input_text= "Argument 1: " + dataset[i]['Arg1RawText'] + ". \nArgument 2: " + dataset[i]['Arg2RawText'] + ". \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\nAnswer:"
## ChatGPT_Prompt Template for discourse relation (PDTB second level)
#         input_text= "Argument 1: " + dataset[i]['Arg1RawText'] + ". \nArgument 2: " + dataset[i]['Arg2RawText'] + ". \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Concession \nB. Contrast \nC. Cause \nD. Pragmatic Cause \nE. Alternative \nF. Conjunction \nG. Instantiation \nH. List \nI. Restatement \nJ. Asynchronous \nK. Synchrony \nAnswer:"

#         input_text= "Argument 1: " + dataset[i]['Arg1RawText'] + ".\nArgument 2: " + dataset[i]['Arg2RawText'] + ".\nConnective between Argument 1 and Argument 2: " + dataset[i]['Conn1'] + "\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison.Concession, if \nB. Comparison.Contrast, however \nC. Contingency.Cause, so \nD. Contingency.Pragmatic, indeed \nE. Expansion.Alternative, instead \nF. Expansion.Conjunction, also \nG. Expansion.Instantiation, for example \nH. Expansion.List, and \nI. Expansion.Restatement, specifically \nJ. Temporal.Asynchronous, before \nK. Temporal.Synchrony, when \nAnswer:"

## (Prompt Engineering) Implict Discourse Relation with ICL Prompt 
#   incontext_prompt="All answer select from following:\nA. Comparison.Concession, nonetheless \nB. Comparison.Contrast, however\nC. Contingency.Cause, so\nD. Contingency.Pragmatic Cause, indeed\nE. Expansion.Alternative, instead\nF. Expansion.Conjunction, also\nG. Expansion.Instantiation, for example\nH. Expansion.List, and\nI. Expansion.Restatement, specifically\nJ. Temporal.Asynchronous, before\nK. Temporal.Synchrony, when\n\nArgument 1: \"Coke could be interested in more quickly developing some of the untapped potential in those markets.\"\nArgument 2: \"A Coke spokesman said he couldn't say whether that is the direction of the talks.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Comparison.Concession, nonetheless\n\nArgument 1: \"Tanks currently are defined as armored vehicles weighing 25 tons or more that carry large guns.\"\nArgument 2: \"The Soviets complicated the issue by offering to include light tanks, which are as light as 10 tons.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Comparison.Contrast, however\n\nArgument 1: \"The male part, the anthers of the plant, and the female, the pistils, of the same plant are within a fraction of an inch or even attached to each other.\"\nArgument 2: \"The anthers in these plants are difficult to clip off.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Contingency.Cause, so\n\nArgument 1: \"I'm not accusing insurers of dereliction of duty, Robert Patricelli of the U.S. Chamber of Commerce told Mr. Waxman's panel.\"\nArgument 2: \"You can't ask one carrier to underwrite on social grounds when that might destroy it in the marketplace.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Contingency.Pragmatic Cause, indeed\n\nArgument 1: \"The judge, suspended from his bench pending his trial, which began this week, vehemently denies all the allegations against him, calling them \"ludicrous\" and \"imaginative, political demagoguery.\" \nArgument 2: \"He blames the indictment on local political feuding, unhappiness with his aggressive efforts to clear the courthouse's docket and a vendetta by state investigators and prosecutors angered by some of his rulings against them.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Alternative, instead\n\nArgument 1: \"Baskets of roses and potted palms adorned his bench.\"\nArgument 2: \"The local American Legion color guard led the way.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Conjunction, also\n\nArgument 1: \"In recent years, demand for hybrid seeds has spurred research at a number of chemical and biotechnology companies, including Monsanto Co., Shell Oil Co. and Eli Lilly & Co.\"\nArgument 2: \"One technique developed by some of these companies involves a chemical spray supposed to kill only a plant's pollen.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Instantiation, for example\n\nArgument 1: \"Accepted bids ranged from 8% to 8.019%.\"\nArgument 2: \"The bank holding company will auction another $50 million in each maturity next Tuesday.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.List, and\n\nArgument 1: \"His recent appearance at the Metropolitan Museum, dubbed \"A Musical Odyssey,\" was a case in point.\"\nArgument 2: \"It felt more like a party, or a highly polished jam session with a few friends, than a classical concert.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Restatement, specifically\n\nArgument 1: \"For example, the chief executive himself now pays 20% of the cost of his health benefits;\"\nArgument 2: \"the company used to pay 100%.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Temporal.Asynchronous, before\n\nArgument 1: \"Panamanian dictator Torrijos, he was told, had granted the shah of Iran asylum in Panama as a favor to Washington.\"\nArgument 2: \"Mr. Sanford was told Mr. Noriega's friend, Mr. Wittgreen, would be handling the shah's security.\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Temporal.Synchrony, when"

## (Prompt Engineering) Explict Discourse Relation
#         input_text= "Argument 1: \"" + dataset[i]['Arg1RawText'] + ".\"\nArgument 2: \"" + dataset[i]['Arg2RawText'] + ".\"\nConnective between Argument 1 and Argument 2: \"" + dataset[i]['Conn1'] + "\"\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison.Concession, if\nB. Comparison.Contrast, however\nC. Contingency.Cause, so\nD. Contingency.Pragmatic Cause, indeed\nE. Expansion.Alternative, instead\nF. Expansion.Conjunction, also\nG. Expansion.Instantiation, for example\nH. Expansion.List, and\nI. Expansion.Restatement, specifically\nJ. Temporal.Asynchronous, before\nK. Temporal.Synchrony, when\nAnswer:"

## (Prompt Engineering) Explict Discourse Relation with ICL Prompt 
#     incontext_prompt="All answer select from following:\nA. Comparison.Concession\nB. Comparison.Contrast\nC. Contingency.Cause\nD. Contingency.Pragmatic Cause\nE. Expansion.Alternative\nF. Expansion.Conjunction\nG. Expansion.Instantiation\nH. Expansion.List\nI. Expansion.Restatement\nJ. Temporal.Asynchronous\nK. Temporal.Synchrony\n\nArgument 1: \"whose hair is thinning and gray and whose face has a perpetual pallor.\"\nArgument 2: \"The prime minister continues to display an energy, a precision of thought and a willingness to say publicly what most other Asian leaders dare say only privately.\" \nConnective between Argument 1 and Argument 2: \"nonetheless\"\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Comparison.Concession\n\nArgument 1: \"they usually give current shareholders the right to buy more stock of their corporation at a large discount if certain events occur.\"\nArgument 2: \"these discount purchase rights may generally be redeemed at a nominal cost by the corporation's directors if they approve of a bidder.\" \nConnective between Argument 1 and Argument 2: \"however\"\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Comparison.Contrast\n\nArgument 1: \"No one could voice an opinion until everybody with more seniority had spoken first.\"\nArgument 2: \"younger employees -- often the most enthusiastic and innovative -- seldom spoke up at all.\"\nConnective between Argument 1 and Argument 2: \"so\"\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Contingency.Cause\n\nArgument 1: \"Republicans have as much interest as Democrats in \"the way the system works.\"\nArgument 2: \"although a majority of Republican lawmakers favor a line-item veto, some, ranging from liberal Oregon Sen. Mark Hatfield to conservative Rep. Edwards are opposed.\"\nConnective between Argument 1 and Argument 2: \"indeed\"\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Contingency.Pragmatic Cause\n\nArgument 1: \"that Warner isn't expected to get any cash in the settlement.\" \nArgument 2: \"Sony is likely to agree to let Warner participate in certain of its businesses, such as the record club of Sony's CBS Records unit.\"\nConnective between Argument 1 and Argument 2: \"instead\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Alternative\n\nArgument 1: \"The South Korean government is signing a protocol today establishing formal diplomatic relations with Poland.\"\nArgument 2: \"The two are signing a trade agreement.\"\nConnective between Argument 1 and Argument 2: \"also\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Conjunction\n\nArgument 1: \"That gives it an enviable strategic advantage, at least until its rivals catch up, but also plenty of managerial headaches.\"\nArgument 2: \"Nissan's U.S. operations include 10 separate subsidiaries -- for manufacturing, sales, design, research, etc. -- that report separately back to Japan.\"\nConnective between Argument 1 and Argument 2: \"for example\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Instantiation\n\nArgument 1: \"We'll raise it through bank loans.We'll raise it through {new} equity.\"\nArgument 2: \"we'll raise it through existing shareholders\" as well as through junk bonds.\"\nConnective between Argument 1 and Argument 2: \"and\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.List\n\nArgument 1: \"to win a contract to design a mapping system for the city of Hiroshima's waterworks.\"\nArgument 2: \"Fujitsu won the right to design the specifications for a computerized system that will show water lines throughout the city.\"\nConnective between Argument 1 and Argument 2: \"specifically\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Expansion.Restatement\n\nArgument 1: \"there were only 70 mental-health clinics in the state.\"\nArgument 2: \"the state of Wisconsin mandated that mental-health care be covered.\"\nConnective between Argument 1 and Argument 2: \"before\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Temporal.Asynchronous\n\nArgument 1: \"I find it hard to ignore our environmental problems.\"\nArgument 2: \"I start my commute to work with eyes tearing and head aching from the polluted air.\"\nConnective between Argument 1 and Argument 2: \"when\" \nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nAnswer: Temporal.Synchrony"