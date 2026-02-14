import os
import json
import argparse
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# ------------------------------------------------------------------------
# 1. System Prompt & Transformation Logic
# ------------------------------------------------------------------------
SYSTEM_PROMPT = """
            You are an **expert-level examiner** with deep expertise in designing **highly challenging and conceptually rigorous multiple-choice questions (MCQs)** for the **Quantitative Aptitude and Analytical Reasoning** sections of top-tier competitive exams.
            Think step by step to generate the question and solve the same, but only output the final answer. Do not show your thinking process.
            **Please DO NOT reveal the solution steps or any intermediate reasoning.**
            """

def load_and_format_data(file_path, limit=None):
    """
    Loads the JSON data and formats it into the chat structure.
    """
    print(f"Reading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply limit if specified
    if limit:
        print(f"⚠️ TEST MODE: Limiting to first {limit} rows only!")
        data = data[:limit]

    formatted_data = []
    
    for entry in data:
        topic = entry.get("topic", "General")
        topics_map = {"Seating Task": "Puzzles/Seating Arrangements (Linear, Circular)", "Syllogisms": "Logical Reasoning/Syllogisms","Blood Relation": "Blood Relations and Family Tree/Family tree logic", "Sequence Tasks":"Series and Patterns/Mixed Series (Alphanumeric)"}
        if topic not in topics_map:
            continue
        topic = topics_map.get(topic)
        # 1. Construct the User Request (The Prompt)
        tmpl = (
                "Generate an EXTREMELY DIFFICULT MCQ on topic: {0}.\n\n"
                "**CRITICAL REQUIREMENTS:**\n"
                '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
                "2.  **Question Quality**: The question must be EXTREMELY DIFFICULT, clear, and test deep conceptual understanding. Avoid trivial or ambiguous questions.\n"
                '3.  **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".\n'
                "4.  **Single Correct Answer**: Ensure that option {2} is only factually correct.\n"
                "5.  **Plausible Distractors**: While option {3} are three incorrect UNIQUE choices which are highly plausible and common misconceptions related to the topic, designed to mislead someone without expert knowledge.\n"
                '6.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
                '7.  **Explanation**: The "explanation" field provides a concise (under 100 words) and clear justification for why the correct answer is correct.\n\n'
                "{5}"
                "RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n"
                "EXAMPLE: {6}\n"
                "{{\n"
                '  "topic": "{7}",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "{8}",\n'
                '  "explanation": "Provide a brief explanation why {9} is correct within 100 words."\n'
                "}}"
            )
        correct_option = entry.get("answer")
        distractors = ", ".join(
            [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
        )
        inc_samples_ex = '''
        Example 1:
            {
                "topic": "Mixed Series (Alphanumeric)",
                "question": "Identify the pattern and select the correct alternative to fill in the blanks: _ c a _ c _ b b a _ c _",
                "choices": [
                    "A) b a c a b",
                    "B) b a a b c",
                    "C) a b c c a",
                    "D) b c a b a"
                ],
                "answer": "B",
                "explanation": "The series follows a cyclic letter shifting and doubling pattern: 'bcaa', 'cabb', 'abcc'. \nLogic: Take the group of three letters (abc), shift the first letter to the end, and double it.\n1. Start: 'abc' -> shift 'a' to end -> 'bca' -> double last -> 'bcaa'.\n2. Next: Start with 'bca' -> shift 'b' to end -> 'cab' -> double last -> 'cabb'.\n3. Next: Start with 'cab' -> shift 'c' to end -> 'abc' -> double last -> 'abcc'."
            }
            Example 2:
            {
                "topic": "Seating Arrangements (Square Table)",
                "question": "Eight friends (P, Q, R, S, T, U, V, W) sit around a square table. Four sit at the corners facing the center, while four sit in the middle of the sides facing outward. \n1. P sits at a corner. \n2. Q sits third to the right of P. \n3. R is an immediate neighbor of Q. \n4. S sits opposite the person who is to the immediate left of R. \n5. T sits second to the right of S. \n6. U is not a neighbor of P or T. \n7. W does not face the center. \n8. V sits to the immediate right of W. \n\nWhat is the position of W with respect to P?",
                "choices": [
                    "A) Third to the left",
                    "B) Immediate right",
                    "C) Third to the right",
                    "D) Fourth to the left"
                ],
                "answer": "C",
                "explanation": "Arrangement logic: \n1. P (Corner, In). Q is 3rd right (Counter-Clockwise) -> Q is at a Side (Out).\n2. R is neighbor of Q. If R is at the corner clockwise to Q, the logic forces contradictions later. Valid scenario: R is at the corner counter-clockwise to Q.\n3. Following the clues: P(1, In), T(2, Out), R(3, In), Q(4, Out), U(5, In), W(6, Out), V(7, In), S(8, Out).\n4. W is at position 6, P is at position 1.\n5. Counting from P (facing In): Right is 8, 7, 6. W is 3rd to the right."
            }
            Example 3:
            {
                "topic": "Syllogisms",
                "question": "Statement I: Only a few Circuits are Batteries.\nStatement II: No Battery is a Diode.\nStatement III: All Diodes are Switches.\n\nConclusion I: All Circuits can never be Diodes.\nConclusion II: Some Switches are definitely not Batteries.\nConclusion III: All Batteries being Switches is a possibility.",
                "choices": [
                    "A) Only Conclusion I follows",
                    "B) Only Conclusions I and II follow",
                    "C) Only Conclusions I and III follow",
                    "D) All Conclusions I, II and III follow"
                ],
                "answer": "D",
                "explanation": "1. Conclusion I follows: 'Only a few Circuits are Batteries' implies some Circuits are Batteries. Since No Battery is a Diode, the part of Circuits that is a Battery cannot be a Diode. Thus, the whole of Circuits cannot be Diodes.\n2. Conclusion II follows: All Diodes are Switches. No Battery is a Diode. Therefore, the part of Switches that are Diodes cannot be Batteries.\n3. Conclusion III follows: There is no negative restriction between Batteries and Switches (only between Batteries and Diodes). Thus, all Batteries could theoretically be placed inside the Switch circle (outside the Diode part)."
            }
        ]
        '''
        user_content = tmpl.format(
            topic,
            topic,
            correct_option,
            distractors,
            correct_option,
            inc_samples_ex,
            topic,
            topic.split("/")[-1],
            correct_option,
            correct_option,
        )

        # 2. Construct the Assistant Response (The Target)
        # Mapping 'answer' from source to 'expected_answer' for target
        assistant_json = {
            "topic": topic,
            "question": entry.get("question"),
            "choices": entry.get("choices"),
            "answer": entry.get("answer"), 
            "explanation": entry.get("explanation")
        }
        
        assistant_content = json.dumps(assistant_json, indent=4)

        # 3. Create the conversation turn
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        formatted_data.append({"messages": messages})

    return Dataset.from_list(formatted_data)

# ------------------------------------------------------------------------
# 2. Main Execution
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen 2.5 for Question Generation')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows for testing')
    args = parser.parse_args()

    # Load Configuration
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config.get('data', {})

    # 1. Load Data
    input_file = data_cfg.get('input_file', 'combined_questions.json')
    full_dataset = load_and_format_data(input_file, limit=args.limit)

    # Split Data (Handle small datasets for testing)
    if args.limit and args.limit < 20:
        print("Dataset too small for 90/10 split. Using 50/50 split for test.")
        split_dataset = full_dataset.train_test_split(test_size=0.5, seed=42)
    else:
        split_dataset = full_dataset.train_test_split(
            test_size=data_cfg.get('val_ratio', 0.05),
            seed=data_cfg.get('seed', 42)
        )
    
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    # 2. Load Model
    print(f"\nLoading model: {model_cfg['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_cfg["base_model"],
        max_seq_length = model_cfg["max_seq_length"],
        dtype = None, # Auto-detect
        load_in_4bit = model_cfg.get("load_in_4bit", True),
    )

    # 3. Apply LoRA Adapters
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(model, **model_cfg["lora"])

    # 4. Setup Tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5", 
        mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
    )

    # 5. Format Prompts
    print("Formatting prompts...")
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return {"text": texts}

    # Reduce num_proc for small tests
    proc_count = 2 if (args.limit and args.limit < 100) else 4

    train_dataset = train_dataset.map(formatting_prompts_func, batched = True, num_proc=proc_count)
    val_dataset = val_dataset.map(formatting_prompts_func, batched = True, num_proc=proc_count)

    print(train_dataset[0])

    # 6. Initialize Trainer
    # Handle auto-detection of bf16/fp16 if not explicitly set in config
    if "bf16" not in train_cfg and "fp16" not in train_cfg:
        train_cfg["bf16"] = is_bfloat16_supported()
        train_cfg["fp16"] = not is_bfloat16_supported()
    
    # Sync max_seq_length
    if "max_seq_length" not in train_cfg:
        train_cfg["max_seq_length"] = model_cfg["max_seq_length"]

    # Override settings for Limit/Test mode
    if args.limit and args.limit < 100:
        train_cfg["logging_steps"] = 1
        train_cfg["save_steps"] = 5
        train_cfg["eval_steps"] = 5
        train_cfg["num_train_epochs"] = 1
        train_cfg["max_steps"] = -1 # Use epochs instead
        train_cfg["eval_strategy"] = "steps"
        train_cfg["save_strategy"] = "steps"

    training_args = SFTConfig(**train_cfg)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = model_cfg["max_seq_length"],
        args = training_args,
    )

    # 7. Configure Response-Only Training
    # This ensures the model learns to output the JSON, not the User Prompt
    print("\nConfiguring Qwen response-only training...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    # 8. Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    trainer_stats = trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint", False))
    print("\n✅ Training complete!")

    # 9. Save
    output_dir = train_cfg.get('output_dir', 'qwen_finetuned')
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
