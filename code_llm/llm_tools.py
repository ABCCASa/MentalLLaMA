import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import re
import json
from typing import Type, List,Dict

class LLM:
    def __init__(self, model_path, device: str, cache_dir = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model.generation_config is not None and self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def call_llm_with_text(self, text: str, max_length) -> str:
        response = self.call_llm([{"role": "user", "content": text}], max_length)
        return response

    def call_llm_with_text_batch(self, text_batch, max_length)-> List[str]:
        messages_batch = [[{"role": "user", "content": text}]for text in text_batch]
        return  self.call_llm_batch(messages_batch, max_length)

    def call_llm_batch(self, messages_batch: List[List[Dict]], max_length) -> List[str]:
        messages_batch = self.tokenizer.apply_chat_template(messages_batch, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(messages_batch, return_tensors="pt", padding=True, padding_side='left').to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_length)
        output_ids = generated_ids[:, len(model_inputs.input_ids[0]):]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = [response.split("</think>")[-1].strip() for response in responses]

        return responses

    def call_llm(self, messages:List[Dict], max_length: int) -> str:
        messages = messages.copy()
        text  = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(text , return_tensors="pt", padding=True).to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_length)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        content = content.split("</think>")[-1] # ignore the thinking step in Qwen3 model
        content = content.strip()
        return content


    def structured_valid(self, response, structure: Type[BaseModel]):
        try:
            matches = re.findall(r"```json(.*?)```", response, re.DOTALL)
            if len(matches) <= 0:
                raise Exception("No JSON block found. Expected a fenced block: ```json ... ```")
            match = matches[-1]
            obj = json.loads(match)
            obj = structure.model_validate(obj).model_dump()
            return True, obj
        except Exception as e:
            output = ""
            for e in str(e).split("\n"):
                if not e.strip().startswith("For further information visit"):
                    output += f"{e}\n"
            return False, output

    def batch_structured_output(self, contents, max_length, structure: Type[BaseModel], batch_size, retry_count=0, print_fre = 5):
        output = [None]*len(contents)
        counts = [-1]*len(contents)
        # build init prompt
        all_messages = []
        for content_i in range(len(contents)):
            content = contents[content_i]
            query = (
                f"Try to extract and summary the information from user provided content and convert them into JSON that matches the given schema:\n"
                f"```json\n{json.dumps(structure.model_json_schema())}\n```. \n"
                f"Make sure to wrap the answer in ```json and ``` tags.\n"
                f"Content:\n {content}")
            all_messages.append([content_i, [{"role": "user", "content": query}], 0])

        for try_count in range(retry_count+1):
            retry_messages = []
            for batch_index in range(0, len(all_messages), batch_size):
                message_batch = all_messages[batch_index: min(batch_index + batch_size, len(all_messages))]
                responses = self.call_llm_batch([m[1] for m in message_batch], max_length)

                # validate
                for i in range(len(responses) - 1, -1, -1):
                    response = responses[i]
                    message_and_id = message_batch[i]
                    content_id = message_and_id[0]
                    messages = message_and_id[1]
                    messages.append({"role": "assistant", "content": response})
                    valid, data = self.structured_valid(response, structure)
                    if valid:
                        output[content_id] = data
                        counts[content_id] = try_count+1
                        if batch_index% (print_fre * batch_size) == 0 and i == 0:
                            print(f"[retry: {try_count}, i: {batch_index}/{len(all_messages)}]")
                            print(f"[query]{contents[content_id].strip()}")
                            print(f"[json]{output[content_id]}\n")
                    else:
                        new_query = f"Failed to extract JSON output, Exception: {data}\n Please retry again."
                        messages.append({"role": "user", "content": new_query})
                        retry_messages.append(message_and_id)
                        if batch_index % (print_fre * batch_size) == 0 and i == 0:
                            print(f"[retry: {try_count}, i: {batch_index}/{len(all_messages)}]")
                            print(f"[query]{contents[content_id].strip()}")
                            print(f"[error]{data.strip()}\n")

            all_messages = retry_messages

        print(len(all_messages),"data fail to structure")
        return output, counts
