"""Load project configuration from configs/config.yaml and secrets from .env.

Usage:
    from src.config import load_config
    cfg = load_config()
"""
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    load_dotenv(ROOT / ".env")  # pulls secrets into os.environ if .env exists
    cfg_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # secrets are read from the environment, never stored in the yaml
    cfg["secrets"] = {
        "email_user": os.getenv("EMAIL_USER"),
        "email_app_password": os.getenv("EMAIL_APP_PASSWORD"),
        "alert_recipient": os.getenv("ALERT_RECIPIENT"),
        "twilio_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "twilio_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "twilio_from": os.getenv("TWILIO_WHATSAPP_FROM"),
        "twilio_to": os.getenv("TWILIO_WHATSAPP_TO"),
    }
    return cfg


if __name__ == "__main__":
    import json
    c = load_config()
    redacted = {k: v for k, v in c.items() if k != "secrets"}
    print(json.dumps(redacted, indent=2))
    print("secrets present:", {k: bool(v) for k, v in c["secrets"].items()})
