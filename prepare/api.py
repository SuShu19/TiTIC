import json
import random
import requests
import config
from prepare.utils import file_opt



def read_token():
    token_path = config.code_path + "/resource/tokens.txt"
    token_file = open(token_path, 'r')
    tokens = token_file.readlines()
    return tokens


def get_api_json(url: object, full_repo: object, type: object, page: object = None, pre_content: object = None) -> object:
    tokens = read_token()
    file_path = config.data_path + "/" + full_repo + "/" + type + ".json"
    if pre_content is not None:
        api_return = pre_content
    else:
        api_return = []
    if page is not None:
        page_cnt = page
    else:
        page_cnt = 0
    while True:
        page_cnt += 1
        if type == 'pulls' or type == 'issues':
            url_ = url + "?page=%s&per_page=100&state=all" % page_cnt
        else:
            url_ = url + "?page=%s&per_page=100" % page_cnt
        print(url_)
        token = tokens[random.randint(0, len(tokens) - 1)].strip()
        headers = {"Authorization": "token %s" % token}
        error_cnt = 0
        while True:
            try:
                response = requests.get(url_,  headers=headers, stream=True)
                if response.status_code == 200:
                    content = json.loads(response.content)
                    break
                else:
                    error_cnt += 1
                    print(response.status_code, response.content, error_cnt)
                    if error_cnt == 1:
                        content = []
                        break
            except Exception as e:
                print(e)
        if isinstance(content, dict):
            content = [content]
        api_return += content
        if content == []:
            break
        if len(content) < 100:      # 取到所有api数据后停止爬虫
            if type == 'pulls' or type == 'issues':
                file_opt.save_json(file_path, api_return)
            break
        if page_cnt % 10 == 0 and (type == 'pulls' or type == 'issues'):
            file_opt.save_json(file_path, api_return)
        else:
            pass

    return api_return



