"""
config_loader.py

This file loads configuration data from config.json for the AI tutor.

"""

import json

def read_config_json(file="config.json"):
    """Read configuration from JSON file using json module"""
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def get_translations():
    """Get translation dictionaries from config"""
    config_data = read_config_json()
    return config_data["translations"]

def get_system_prompt():
    """Get the system prompt from config"""
    config_data = read_config_json()
    return config_data["system_prompt"]

def get_emoji_regex_pattern():
    """Return a compiled emoji regex pattern from config"""
    config_data = read_config_json()
    emoji_patterns = config_data["emoji_patterns"]
    parts = list(emoji_patterns.values())
    return "[" + "".join(parts) + "]+"

def get_status_messages():
    """Get status messages for different languages from config"""
    config_data = read_config_json()
    return config_data["status_messages"]

def get_status_message(key, lang="en"):
    """Get a specific status message in the specified language"""
    status_messages = get_status_messages()
    lang_messages = status_messages.get(lang, status_messages.get("en"))
    return lang_messages.get(key, key)

def get_urdu_numbers():
    """Get English to Urdu number mapping from config"""
    config_data = read_config_json()
    return config_data["urdu_numbers"]

def get_tts_patterns():
    """Get regex patterns for TTS cleanup from config"""
    config_data = read_config_json()
    return config_data["tts_patterns"]

def get_tts_symbols_map():
    """Get TTS symbol mappings from config"""
    config_data = read_config_json()
    return config_data["tts_symbols_map"]

def get_model_registry():
    """Get model registry from config"""
    config_data = read_config_json()
    return config_data["model_registry"]

def get_config_value(key):
    """Get any configuration value by key"""
    config_data = read_config_json()
    return config_data[key]

def get_grade_options():
    """Return grade options using for labels and values for the dropdown"""
    return [{"label": str(i), "value": str(i)} for i in range(1, 13)]

def get_subject_options(lang="en"):
    """Return subject dropdown options by language from translations.
       Returns list of dicts: {label, value}. Value remains in English for LLM.
    """
    config_data = read_config_json()
    translations = config_data.get("translations")
    lang_translations = translations.get(lang)
    subjects = lang_translations.get("subject_options")
    return subjects

def get_user_profile():
    """Get the saved user profile for language, grade and subject"""
    config_data = read_config_json()
    return config_data.get("user_profile")

def save_user_profile(language=None, grade=None, subject=None):
    """Save user profile settings to config.json
       Recieves the optional language, grade and subject values
    """
    config_data = read_config_json()
    
    # Update only provided values
    if language is not None:
        config_data["user_profile"]["language"] = language
    if grade is not None:
        config_data["user_profile"]["grade"] = str(grade)
    if subject is not None:
        config_data["user_profile"]["subject"] = subject
        
    # Write back to file
    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    
    return True
