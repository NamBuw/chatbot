import os, json

USER_INFO_FILE = 'user_info.json'

def load_user_info():
    try:
        if os.path.exists(USER_INFO_FILE):
            with open(USER_INFO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print("Error loading JSON:", e)
        return {}

if __name__ == '__main__':
    info = load_user_info()
    print("User info:", info)
