"""
Configuration setup using Dynaconf.
"""
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",  # Export envvars with `export DYNACONF_FOO=bar`.
    settings_files=[
        "settings/settings.toml",  # Main settings file
        "settings/.secrets.toml",  # Sensitive data
    ],
    environments=True,  # Enable multiple environments
    load_dotenv=True,  # Load environment variables from .env file
    env_switcher="ENV_FOR_DYNACONF",  # Environment variable to switch environments
    default_env="default",  # Default environment
) 