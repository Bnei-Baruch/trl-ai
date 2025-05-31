"""
Script for preparing translation data from PostgreSQL and saving to files.
"""

import logging
import json
import os
from time import sleep
from typing import List, Dict, Any
import psycopg2
from config import settings
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# SQL query for fetching file translations
query_files = """
SELECT f.uid as en_uid, f2.uid as he_uid, cu.uid as cu_uid
FROM files f
INNER JOIN content_units cu ON cu.id = f.content_unit_id
INNER JOIN files f2 ON f2.content_unit_id = cu.id
WHERE f2.language = 'he'
    AND f2.type = 'text'
    AND f2.properties->>'insert_type' = 'tamlil'
    AND f.language = 'en'
    AND f.type = 'text'
    AND f.properties->>'insert_type' = 'tamlil'
OFFSET %s
LIMIT %s
"""

# Add this after the query_files definition:

query_count = """
SELECT COUNT(*)
FROM files f
INNER JOIN content_units cu ON cu.id = f.content_unit_id
INNER JOIN files f2 ON f2.content_unit_id = cu.id
WHERE f2.language = 'he'
    AND f2.type = 'text'
    AND f2.properties->>'insert_type' = 'tamlil'
    AND f.language = 'en'
    AND f.type = 'text'
    AND f.properties->>'insert_type' = 'tamlil'
"""

# Get database settings with defaults
db_settings = settings.database


def get_total_count(conn) -> int:
    """Get total number of translation pairs available."""
    with conn.cursor() as cur:
        cur.execute(query_count)
        result = cur.fetchone()
        return result[0] if result else 0


def get_db_connection():
    """Create a database connection."""
    try:
        conn = psycopg2.connect(
            host=db_settings.host,
            port=db_settings.port,
            database=db_settings.name,
            user=db_settings.user,
            password=settings.database_password,
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def fetch_trl_item(conn, batch_size, offset) -> List[Dict[str, Any]]:
    """Fetch translations from database in batches."""
    translations = []
    with conn.cursor() as cur:
        cur.execute(query_files, (offset, batch_size))
        rows = cur.fetchall()

        if rows:
            logger.info(f"Fetched {len(rows)} rows for offset {offset}")

            for row in rows:
                en_text = fetch_files_by_uid(row[0])
                he_text = fetch_files_by_uid(row[1])
                cu_uid = fetch_files_by_uid(row[2])
                translation = {
                    "translation": {
                        "en": en_text,
                        "he": he_text,
                        "cu_uid": cu_uid,
                    }
                }
                translations.append(translation)

            logger.info(
                f"Processed {len(translations)} translations for offset {offset}"
            )

    return translations


def fetch_files_by_uid(uid: str):
    url = f"https://kabbalahmedia.info/assets/api/doc2text/{uid}"
    response = requests.get(url)
    return response.text


def initialize_output_file(output_file: str):
    """Initialize the output JSON file with an empty translations list."""
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"translations": []}, f)

    logger.info(f"Initialized output file: {output_file}")


def append_translations_to_file(translations: List[Dict[str, Any]], output_file: str):
    """Append translations to the existing JSON file."""
    try:
        # Read existing content
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Append new translations
        data["translations"].extend(translations)

        # Write back to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Appended {len(translations)} translations to {output_file}")
    except Exception as e:
        logger.error(f"Error appending translations to file: {e}")
        raise


def prepare_data(batch_size):
    """Main function to fetch translation data and save to a single file."""
    # Use default output directory if not specified in settings
    output_dir = os.path.join(
        getattr(settings.training, "output_dir", "./outputs"), "data"
    )
    output_file = os.path.join(output_dir, "translations.json")

    try:
        # Initialize output file
        initialize_output_file(output_file)

        # Connect to database
        conn = get_db_connection()

        batch_num = 0
        max_batches = settings.database.max_batch_size
        if max_batches is None:
            max_batches = get_total_count(conn) % batch_size

        while True:
            if batch_num >= max_batches:
                logger.info(f"Reached maximum number of batches ({max_batches})")
                break

            offset = batch_num * batch_size
            translations = fetch_trl_item(conn, batch_size=batch_size, offset=offset)

            if not translations:
                logger.info("No more translations to fetch")
                break

            # Append translations to file
            append_translations_to_file(translations, output_file)

            batch_num += 1
            sleep(5)

        logger.info("Data preparation completed successfully")

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise
    finally:
        if "conn" in locals():
            conn.close()


def get_translation_sample(limit: int = 5):
    """Get a sample of translations from the saved file."""
    output_dir = os.path.join(settings.training.output_dir, "data")
    file_path = os.path.join(output_dir, "translations.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["translations"][:limit]
    except FileNotFoundError:
        logger.error(f"Translation file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading translation file: {e}")
        return []


if __name__ == "__main__":
    batch_size = getattr(db_settings, "batch_size", 1000)
    prepare_data(batch_size)
