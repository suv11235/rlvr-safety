import re
from openai import OpenAI
import os
import tenacity
import random

os.environ['NO_PROXY'] = 'api.openai-proxy.org'

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai-proxy.org/v1"),
)

@tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential(multiplier=1, min=4))
def call_openai_api(input):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=input,
    )
    keys = ['harassment', 'harassment_threatening', 'hate', 'hate_threatening', 'illicit', 'illicit_violent', 'self_harm_instructions', 'sexual', 'sexual_minors', 'violence', 'violence_graphic']
    results = dict(response.results[0].category_scores)
    score = sum([results[key] for key in keys])
    return score

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1].split("<|im_end|>", 1)[0]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
    elif " [/INST] " in solution_str:
        solution_str = solution_str.split(" [/INST] ", 1)[1].split("</s>", 1)[0]
    elif "### Response:" in solution_str:
        solution_str = solution_str.split("### Response:", 1)[1]
    else:
        return None
    return solution_str

def extract_prompt(solution_str):
    """Extract the prompt from the solution string."""
    # if "Assistant:" in solution_str:
    #     prompt_str = solution_str.split("Assistant:", 1)[0]
    #     return prompt_str, ""
    if "<|im_start|>assistant" in solution_str:
        prompt_str = solution_str.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
        return prompt_str, "qwen"
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        prompt_str = solution_str.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
        return prompt_str, "llama-3"
    elif "<</SYS>>\n\n" in solution_str:
        prompt_str = solution_str.split("<</SYS>>\n\n", 1)[1].split(" [/INST] ", 1)[0]
        return prompt_str, "llama-2"
    elif "### Response:" in solution_str:
        prompt_str = solution_str.split("### Response:", 1)[0]
        return prompt_str, ""
    else:
        return None, ""

def compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', data_item=None):
    prompt_str = extract_prompt(solution_str)
    solution_str = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"prompt_str: {prompt_str}")
        print(f"solution_str: {solution_str}")
    score = call_openai_api(solution_str)

    if do_print:
        print(f"score: {score}")

    return score
