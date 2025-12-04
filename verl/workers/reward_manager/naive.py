# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict, deque
import json
import os

class DataTracker:
    """Track data length and scores in a fixed-size queue."""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.data_queue = deque(maxlen=max_size)
        self.steps = 0

    def update_step(self):
        self.steps += 1

    def add_data(self, sequence, length, score):
        """Add new data point (length, score) to the queue."""
        self.data_queue.append({"sequence": sequence, "length": length, "score": score})
    
    def dump_data(self, output_dir):
        """Dump data to a file."""
        with open(f"{output_dir}/data_tracker_{self.steps}.json", "w") as f:
            json.dump(list(self.data_queue), f)
        self.data_queue.clear()


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', track_data=True, tracker_max_size=1000) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # Initialize data tracker
        self.data_tracker = DataTracker(tracker_max_size) if track_data else None

    def __call__(self, data: DataProto, step_index=None, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        self.data_tracker.update_step()
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            # prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            if step_index is None:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    data_item=data_item,
                    extra_info=extra_info,
                )
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    data_item=data_item,
                    step_index=step_index,
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # Track data if tracker is enabled
            if self.data_tracker is not None:
                self.data_tracker.add_data(sequences_str, int(valid_response_length), float(reward))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if self.data_tracker is not None:
            EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
            OUTPUT_DIR = os.getenv("OUTPUT_DIR")
            # if data_tracker directory does not exist, create it
            if not os.path.exists(f"{OUTPUT_DIR}/data_tracker"):
                os.makedirs(f"{OUTPUT_DIR}/data_tracker")
            self.data_tracker.dump_data(f"{OUTPUT_DIR}/data_tracker")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
