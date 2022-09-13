import json

def printFormattedJson(json_data: dict):
    print(json.dumps(json_data, indent=4, ensure_ascii=False))
    print()