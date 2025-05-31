"""
Script to use the fine-tuned NLLB model for translation.
"""

import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from settings.config import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Translate text using NLLB model")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to translate",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="default",
        help="Environment to use (default, development, production)",
    )
    return parser.parse_args()

def translate_text(model, tokenizer, text: str) -> str:
    """Translate text from source language to target language."""
    # Prepare the text for translation
    inputs = tokenizer(text, return_tensors="pt")
    
    # Set the language tokens
    tokenizer.src_lang = settings.training.source_lang
    forced_bos_token_id = tokenizer.lang_code_to_id[settings.training.target_lang]
    
    # Generate translation
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=settings.generation.max_length,
        num_beams=settings.generation.num_beams,
        early_stopping=settings.generation.early_stopping
    )
    
    # Decode the generated tokens
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translation

def main():
    args = parse_args()
    
    # Set environment if specified
    if args.env != "default":
        settings.configure(ENV_FOR_DYNACONF=args.env)
    
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.training.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(settings.training.output_dir)
    
    # Translate the text
    translation = translate_text(
        model,
        tokenizer,
        args.text
    )
    
    print(f"\nSource ({settings.training.source_lang}): {args.text}")
    print(f"Translation ({settings.training.target_lang}): {translation}")

if __name__ == "__main__":
    main() 