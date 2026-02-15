import time
import torch
import re
import json  # Added for pre-validation
from typing import Optional, List, Union, Tuple
from unsloth import FastLanguageModel

# Set seed for reproducibility
torch.manual_seed(3407)

class AAgent(object):
    def __init__(self, **kwargs):
        # 1. Load Qwen 2.5 14B Instruct using Unsloth
        # Using local path as per your setup
        model_name = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        # model_name = "hf_models/qwen_cot_merged_14b"
        max_seq_length = 4096 
        dtype = None 
        load_in_4bit = False

        print(f"Loading model: {model_name}...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
            )
        except Exception as e:
            print(f"Local load failed: {e}")
            print("Loading from Unsloth Hub...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
            )
        
        FastLanguageModel.for_inference(self.model)
        self.tokenizer.padding_side = "left"

    def _clean_response(self, text: str) -> str:
        """
        Helper to remove markdown fences and aggressively extract JSON.
        Uses Regex to find the first valid JSON object boundaries.
        """
        # 1. Basic clean of markdown
        text = text.replace("```json", "").replace("```", "").strip()

        # 2. Regex Extraction: Look for content between first { and last }
        # [\s\S]* matches any character including newlines
        match = re.search(r"\{[\s\S]*\}", text)
        
        if match:
            candidate = match.group(0)
            # 3. JSON Pre-validation
            try:
                # If it parses as valid JSON, return the clean candidate
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # If regex found brackets but content isn't valid JSON,
                # fall back to returning the text (stripped of fences)
                # so the AnsweringAgent's retry logic can handle it.
                pass
        
        return text

    def generate_response(
        self, 
        message: Union[str, List[str]], 
        system_prompt: Optional[str] = None, 
        **kwargs
    ) -> Union[str, List[str], Tuple[Union[str, List[str]], int, float]]:
        
        # 1. Handle Input (Unify into list)
        if isinstance(message, str):
            messages_list = [message]
        else:
            messages_list = message

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        # 2. Format Prompts
        formatted_prompts = []
        for msg in messages_list:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(text)

        # 3. Tokenize
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")

        # 4. Generate
        tgps_show_var = kwargs.get("tgps_show", False)
        start_time = time.time() if tgps_show_var else 0

        # Parameters optimized for valid JSON generation
        # (Defaults kept exactly as requested)
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.1) 
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=(temperature > 0),
            )
        
        generation_time = time.time() - start_time if tgps_show_var else 0

        # 5. Decode
        decoded_responses = []
        token_count = 0
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        
        for i, token_ids in enumerate(generated_tokens):
            if tgps_show_var:
                token_count += len(token_ids)
            
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            # CLEAN THE MARKDOWN HERE (with improved regex cleaner)
            text = self._clean_response(text)
            decoded_responses.append(text)

        result = decoded_responses[0] if isinstance(message, str) else decoded_responses

        if tgps_show_var:
            return result, token_count, generation_time
        else:
            return result