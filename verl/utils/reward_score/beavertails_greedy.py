import re
import os
import tenacity
import random
import requests
import torch
import json

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # if "<|assistant|>\n" in solution_str:
    #     solution_str = solution_str.split("<|assistant|>\n", 1)[1].split("<end_of_turn>", 1)[0].split("<|assistant|>", 1)[0]
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1].split("<|im_end|>", 1)[0].split("<|im_start|>", 1)[0].split("<|endoftext|>", 1)[0]
    # elif "<start_of_turn>model" in solution_str:
    #     solution_str = solution_str.split("<start_of_turn>model", 1)[1].split("<end_of_turn>", 1)[0]
    if "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
    elif "[/INST]" in solution_str:
        solution_str = solution_str.split("[/INST]", 1)[1].split("</s>", 1)[0]
    # elif "### Response:" in solution_str:
    #     solution_str = solution_str.split("### Response:", 1)[1]
    # else:
    #     return None
    return solution_str

def extract_prompt(solution_str):
    """Extract the prompt from the solution string."""
    # if "Assistant:" in solution_str:
    #     prompt_str = solution_str.split("Assistant:", 1)[0]
    #     return prompt_str, ""
    if "<|im_start|>assistant" in solution_str:
        prompt_str = solution_str.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
        return prompt_str, "qwen"
    # if "<|assistant|>\n" in solution_str:
    #     prompt_str = solution_str.split("<|user|>\n", 1)[1].split("<|assistant|>", 1)[0]
    #     return prompt_str, "olmo2"
    # elif "<start_of_turn>model" in solution_str:
    #     prompt_str = solution_str.split("<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]
    #     return prompt_str, "gemma2"
    if "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        prompt_str = solution_str.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
        return prompt_str, "llama-3"
    elif "<s>[INST]" in solution_str:
        prompt_str = solution_str.split("<s>[INST]", 1)[1].split("[/INST]", 1)[0]
        return prompt_str, "ministral"
    # elif "### Response:" in solution_str:
    #     prompt_str = solution_str.split("### Response:", 1)[0]
    #     return prompt_str, ""
    else:
        return None, ""


def compute_score(data_source,
                  solution_str,
                  ground_truth,
                  extra_info,
                  data_item=None,
                  step_index=None,
                  method='strict'):
    if data_item is None:
        return 0.0

    prompt_ids = data_item.batch['prompts']
    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch['responses']
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    old_log_prob = data_item.batch['old_log_probs']
    # compute average log_prob using old_log_prob and response_mask
    old_log_prob = old_log_prob[:valid_response_length]
    old_log_prob_avg = old_log_prob.sum() / valid_response_length
    perplexity = torch.exp(-old_log_prob_avg)
    negative_perplexity = -perplexity
    # score = negative_perplexity.item()

    old_entropy = data_item.batch['old_entropy']
    old_entropy = old_entropy[:valid_response_length]
    old_entropy_avg = old_entropy.sum() / valid_response_length
    negative_entropy = -old_entropy_avg
    score = negative_entropy.item()


    prompt_str, model_type = extract_prompt(solution_str)
    solution_str = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"prompt_str: {prompt_str}")
        print(f"solution_str: {solution_str}")
        print(f"score: {score}")

    return score
