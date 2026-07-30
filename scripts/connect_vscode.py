import json
import os
import sys

def setup_continue():
    config_dir = os.path.expanduser("~/.continue")
    config_path = os.path.join(config_dir, "config.json")
    
    os.makedirs(config_dir, exist_ok=True)
    
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {config_path}. Overwriting.")
            
    if "models" not in config:
        config["models"] = []
        
    # Check if swiftlet is already there
    for model in config["models"]:
        if model.get("title") == "Swiftlet Qwen":
            print("Continue.dev is already configured for Swiftlet.")
            return

    config["models"].insert(0, {
        "title": "Swiftlet Qwen",
        "provider": "openai",
        "model": "qwen",
        "apiBase": "http://localhost:8000/v1",
        "apiKey": "none"
    })
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print("✅ Successfully configured Continue.dev for Swiftlet!")


def setup_cline():
    if sys.platform == "darwin":
        base_dir = os.path.expanduser("~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings")
    elif sys.platform == "win32":
        base_dir = os.path.expanduser("~/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/settings")
    else:
        base_dir = os.path.expanduser("~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings")
        
    os.makedirs(base_dir, exist_ok=True)
    config_path = os.path.join(base_dir, "cline_api_settings.json")
    
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass

    config["apiProvider"] = "openai"
    config["openAiBaseUrl"] = "http://localhost:8000/v1"
    config["openAiApiKey"] = "none"
    config["openAiModelId"] = "qwen"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print("✅ Successfully configured Cline for Swiftlet!")

if __name__ == "__main__":
    print("Connecting VS Code to Swiftlet...")
    setup_continue()
    setup_cline()
    print("\nAll done! Restart VS Code (or reload the window) if it's already open.")
