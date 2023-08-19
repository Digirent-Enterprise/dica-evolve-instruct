import ast
import time
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
import torch
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from transformers.pipelines.pt_utils import KeyDataset
import argparse
from datasets import load_dataset


class Mutation(Enum):
    FRESH_START = 0
    ADD_CONSTRAINTS = 1
    DEEPEN = 2
    CONCRETIZE = 3
    INCREASE_REASONING = 4
    COMPLICATE = 5

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response in Vietnamese that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


class WizardLM:
    def __init__(
            self,
            llm_pipeline: pipeline = None,
            seed_data: List[str] = None,
            column_names: List[str] = ["instruction"],
            num_rows: int = 10,
            min_len_chars: int = 512,
            max_len_chars: int = 1024,
            verbose: bool = False,
    ):
        self.llm_pipeline = llm_pipeline
        self.column_names = column_names
        self.num_rows = num_rows
        self.verbose = verbose
        self.seed_text_list = []
        self.seed_data = seed_data
        self.prompts = []
        self.final_prompts = []
        self.final_answers = []
        self.min_len_bytes = min_len_chars
        self.max_len_bytes = max_len_chars
        self.prompt_templates = dict()
        self.prompt_templates['base'] = ""
        write_in_vietnamese = "Write in Vietnamese."
        self.prompt_translate_into_vietnamese = """
Translate #Given Prompt# to #New Prompt# in Vietnamese. The output must be in Vietnamese"

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.FRESH_START] = \
            self.prompt_templates['base'] + \
            f"""Rewrite #Given Prompt# by switching the locale into Vietnamese and create #New Prompt#.
            {write_in_vietnamese}

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.COMPLICATE] = \
            self.prompt_templates['base'] + \
            f"""Rewrite #Given Prompt# to make it slightly more complicated, and create #New Prompt#.
            {write_in_vietnamese}

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.ADD_CONSTRAINTS] = \
            self.prompt_templates['base'] + \
            f"""Add a few more constraints or requirements to #Given Prompt#,
            and create #New Prompt#. {write_in_vietnamese}

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.DEEPEN] = \
            self.prompt_templates['base'] + \
            f"""Slightly increase the depth and breadth of #Given Prompt#, and create #New Prompt#. {write_in_vietnamese}

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.CONCRETIZE] = \
            self.prompt_templates['base'] + \
            f"""Make #Given Prompt# slightly more concrete, and create #New Prompt#. {write_in_vietnamese}

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.INCREASE_REASONING] = \
            self.prompt_templates['base'] + \
            f"""If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to explicitly
            request multi-step reasoning, and create #New Prompt#. {write_in_vietnamese}

#Given Prompt#:
<PROMPT>
"""

    def run(self):
        self.create_seed_prompts()
        self.create_prompts()
        self.create_answers()
        list_qa = []
        for i in range(len(self.final_prompts)):
            if len(self.final_answers[i]) > 10:
                list_qa.append(
                    {
                        'input': self.final_prompts[i],
                        'output': self.final_answers[i],
                    }
                )
        import json
        import uuid
        with open(f"{self.seed_data.replace('.jsonl', '').replace('json', '')}.%s.json" % str(uuid.uuid4())[:4],
                  "wt") as f:
            f.write(json.dumps(list_qa, indent=2, ensure_ascii=False))

    def create_seed_prompts(self):
        """
        Turn self.seed_data into a list of strings of text self.source_text_list
        Each text string can represent as little as a word, or as much as document.
        Just has to be representative of some concept or body of text.

        :return: None
        """
        import os
        if isinstance(self.seed_data, str) and os.path.exists(self.seed_data):
            data = load_dataset("json", data_files=self.seed_data)
            self.seed_text_list = [d["instruction"] for d in data['train']]
            assert self.seed_text_list, "data import failed, got empty list"


    def create_prompts(self):
        """
        Create prompts using seed data and mutations.
        """
        print("Creating %d prompts." % self.num_rows)
        assert self.seed_text_list, "Must have seed text list"
        t0 = time.time()
        self.prompts.clear()

        for i in range(self.num_rows):
            new_prompt = np.random.choice(self.seed_text_list)
            formatted_prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=new_prompt)
            self.prompts.append(formatted_prompt)
        i = 0
        while self.mutate(i):
            print("Iteration: %d" % i)
            i += 1
        t1 = time.time()
        print("Done creating %d prompts in %.4f seconds." % (len(self.final_prompts), t1 - t0))
        print(self.final_prompts)

    def create_answers(self):
        print("Creating answers for %d prompts." % len(self.final_prompts))
        t0 = time.time()
        ds = self.convert_list_to_dataset(self.final_prompts)
        self.final_answers = self.llm_pipeline(ds['train'])
        t1 = time.time()
        print("Done creating answers for %d prompts in %.4f seconds." %
              (ds['train'].num_rows, t1 - t0))

    def convert_list_to_dataset(self, text_list):
        df = pd.DataFrame({'text': text_list})
        ds = DatasetDict()
        ds['train'] = Dataset.from_pandas(df)
        return ds

    def mutate(self, iteration):
        assert len(self.prompts) == self.num_rows
        list_prompts = []
        mutations = []
        for i in range(self.num_rows):
            mutation = np.random.choice(Mutation)
            mutations.append(mutation)
            # if mutation == Mutation.FRESH_START:
            #     mutation = Mutation.COMPLICATE
            before = self.prompts[i]
            prompt = self.prompt_templates[mutation].replace("<PROMPT>", before)
            list_prompts.append(prompt)

        ds = self.convert_list_to_dataset(list_prompts)
        assert ds['train'].num_rows == len(list_prompts) == self.num_rows == len(self.prompts)
        t0 = time.time()
        after = self.llm_pipeline(ds['train'])
        assert len(after) == self.num_rows

        for i in range(len(after)):
            after[i] = after[i].split("Prompt#:")[-1].strip()
            for pp in ['New Prompt:\n', 'New Prompt: ']:
                if after[i][:len(pp)] == pp:
                    after[i] = after[i][len(pp):]
            after[i] = after[i].strip()
            use_new_prompt, why = self.change_approved(self.prompts[i], after[i])
            if self.verbose:
                print("===========================")
                print("Old Prompt: %s" % self.prompts[i])
                print("Mutation: %s" % mutations[i].name)
                print("New Prompt: %s" % after[i])
                print("===========================")

            if use_new_prompt:
                if self.max_len_bytes >= len(after[i]) >= self.min_len_bytes:
                    self.final_prompts.append(after[i])
                    print("Prompt was accepted, now have %d good prompts." % len(self.final_prompts))
                    self.prompts[i] = np.random.choice(self.seed_text_list)
                    print("Creating new prompt.")
                else:
                    self.prompts[i] = after[i]
                    print("Prompt was successfully modified.")
            else:
                print("Mutation rejected, will try again. Reason: %s" % why)
            print("", flush=True)
        return len(self.final_prompts) < self.num_rows

    def change_approved(self, before, after):
        if before == after:
            return False, "same"
        if after.count('\n') > after.count(" ") * 2:
            return False, "too many lines"
        if after.count('\n') == after.count("- ") > 10:
            return False, "too many items"
        if self.prompt_templates['base'] and self.prompt_templates['base'] in after:
            return False, "prompt leaked 1"
        if "#New Prompt#" in after:
            return False, "prompt leaked 2"
        if "new prompt" in after.lower():
            return False, "prompt leaked 3"
        if "openai" in after.lower():
            return False, "AI"
        if "gpt" in after.lower() and "gpt" not in before.lower():
            return False, "AI"
        if "Tôi xin lỗi" in after.lower() and "Xin lỗi" not in before.lower() and len(after) < len(before):
            return False, "sorry"
        if False:
            # too slow in general, not needed
            prompt = """Are the two following prompts equal to each other?
To be equal, they must meet two requirements:
1. Both prompts have the same constraints and requirements.
2. Both prompts have the same depth and breath of the inquiry.
First prompt: %s
Second prompt: %s
Answer with 'Equal' or 'Not Equal'. No need to explain the reason.""" % (before, after)
            answer = self.llm_pipeline(prompt)
            if 'not equal' not in answer.lower():
                return False, "equal"
        return True, "ok"


class HFPipeline:
    def __init__(self, model, max_new_tokens=None, batch_size=None, **kwargs):
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model, padding_side="left", device_map=3, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("-----------loading model")
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'
        config.init_device = 'cuda:3'

        print(config)

        model_obj = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map=3, trust_remote_code=True, config=config, load_in_4bit=True)
        del model_obj

        print("-----------loading pipeline")
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=3,
            **kwargs,
        )
        print("loading pipeline done.")
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def __call__(self, dataset):
        """
        Passes dataset to LLM and returns the responses.
        :param dataset:  Hugging Face dataset containing a 'text' column with prompts.
        :return: list of strings with responses.
        """

        ret = []
        for i, out in enumerate(tqdm(
            self.pipeline(
                KeyDataset(dataset, "text"),
                max_new_tokens=self.max_new_tokens,
                batch_size=self.batch_size,
            )
        )):
            # remove input in case pipeline is using completion/plain prompt
            response = out[0]["generated_text"]
            response = response.replace(dataset[i]['text'], '').strip()
            ret.append(response)
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument("--seed_file", type=str)
    parser.add_argument("--column_names", nargs='+', default="instruction")
    parser.add_argument("--num_rows", type=int, default=5)
    parser.add_argument("--min_len_chars", type=int, default=32)
    parser.add_argument("--max_len_chars", type=int, default=512)

    args = parser.parse_args()

    llm_pipeline = HFPipeline(
        "mosaicml/mpt-7b-instruct",
        max_new_tokens=1000,
        do_sample=True,
        batch_size=8,
        load_in_4bit=True
    )

    wizardlm = WizardLM(
        llm_pipeline=llm_pipeline,
        seed_data=args.seed_file,
        column_names=args.column_names,
        num_rows=args.num_rows,
        min_len_chars=args.min_len_chars,
        max_len_chars=args.max_len_chars,
        verbose=True,
    )
    wizardlm.run()

# CUDA_VISIBLE_DEVICES=3 python mpt-7b/evolve.py --seed_file seed_data.json --column_names instruction input --num_rows 20

# def test_check():
#     import pickle
#     with open("prompts.pickle", "rb") as f:
#         X = pickle.loads(f.read())
#         print(X)
