import os
import config
from prepare import api
from prepare.utils import file_opt, initialize
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

def get_full_pr_issue(full_repo):
    print("start " + full_repo)
    if os.path.exists(config.data_path + "/" + full_repo):
        pass
    else:
        os.makedirs(config.data_path + "/" + full_repo)
    types = ['issues']     # 必须先执行pr的操作
    pulls_api = []
    for type in types:
        break_page = None
        file_path = config.data_path + "/" + full_repo + "/" + type + ".json"
        if os.path.exists(file_path):
            pre_pull_is = file_opt.read_json(file_path)
            break_page = int(len(pre_pull_is) / 100)
        else:
            pre_pull_is = None
        type_url = "https://api.github.com/repos/" + full_repo + "/" + type
        pulls_api.append({"type": type,
                          "list": api.get_api_json(type_url, full_repo, type=type, page=break_page,
                                                   pre_content=pre_pull_is)})
    return pulls_api

if __name__ == '__main__':
    repo_list = initialize.repo_list
    with PoolExecutor(max_workers=1) as executor:
        for _ in executor.map(get_full_pr_issue, repo_list):
            pass