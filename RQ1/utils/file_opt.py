import json

def save_json(file_path, content):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(content, f)
    return None

def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        content = f.readlines()
        for i in range(len(content)):
            content[i] = content[i].strip()
    return content

def save_txt(file_path, content):
    with open(file_path, 'w', encoding='utf8') as f:
        for i in content:
            f.write(i + "\n")
    return None



