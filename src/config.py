"""
Configuration setup using Dynaconf.
"""

import os
from pathlib import Path
from dynaconf import Dynaconf

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

settings = Dynaconf(
    envvar_prefix="DYNACONF",  # Export envvars with `export DYNACONF_FOO=bar`.
    settings_files=[
        os.path.join(ROOT_DIR, "settings", "settings.toml"),  # Main settings file
        os.path.join(ROOT_DIR, "settings", ".secrets.toml"),  # Sensitive data
    ],
    environments=True,  # Enable multiple environments
    load_dotenv=True,  # Load environment variables from .env file
    env_switcher="ENV_FOR_DYNACONF",  # Environment variable to switch environments
    default_env="default",  # Default environment
)
