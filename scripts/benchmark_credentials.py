"""Optional solver summary transport routing; judges resolve their own profile."""
import os
from pathlib import Path
from dotenv import load_dotenv


def configure(profile=None):
    if not profile or not profile.endswith('_openrouter'):
        return
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('OPENROUTER_API_KEY is required for summary transport')
    os.environ['SUMMARY_LLM_API_KEY'] = key
    os.environ['SUMMARY_LLM_BASE_URL'] = os.environ.get('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1'
