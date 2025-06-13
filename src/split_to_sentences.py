import json
import nltk
import re
from pathlib import Path
import logging

nltk.download("punkt")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_and_pair(data):
    en = data["en"]
    he = data["he"]

    en_sents = split_sentences_en(en)
    he_sents = split_sentences_he(he)

    sentence_pairs = []
    min_len = min(len(en_sents), len(he_sents))
    logger.info(f"Splitting into {min_len} sentence pairs.")
    for i in range(min_len):
        logger.debug(f"Pair {i+1}: EN: {en_sents[i]} | HE: {he_sents[i]}")
        sentence_pairs.append({"en": en_sents[i], "he": he_sents[i]})
    return sentence_pairs


def load_sentence_pairs(input_path):
    logger.info(f"Loading translations from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['translations'])} translation documents.")
    return data["translations"]


def save_sentence_pairs(sentence_pairs, output_path):
    logger.info(f"Saving {len(sentence_pairs)} sentence pairs to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"translations": sentence_pairs}, f, ensure_ascii=False, indent=2)
    logger.info("Save complete.")


def split_sentences_en(text):
    return nltk.sent_tokenize(text)


def split_sentences_he(text):
    # Simple split by period, question mark, exclamation mark
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


if __name__ == "__main__":
    input_file = "outputs/data/translations.json"
    output_file = "outputs/data/sentence_pairs.json"
    data = load_sentence_pairs(input_file)
    logger.info("Starting sentence splitting and pairing...")
    sentence_pairs = []
    for idx, item in enumerate(data):
        logger.info(f"Processing document {idx+1}/{len(data)}")
        sentence_pairs.extend(split_and_pair(item["translation"]))
    save_sentence_pairs(sentence_pairs, output_file)
    print(f"Saved sentence-level pairs to {output_file}")
