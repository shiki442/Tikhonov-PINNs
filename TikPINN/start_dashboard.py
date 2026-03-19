import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna_dashboard import run_server

# 1. 这里的路径替换成你实际的 .log 文件路径
log_file_path = "./study.log" 
storage = JournalStorage(JournalFileStorage(log_file_path))

# 2. 启动可视化 Web 服务
print(f"正在读取 {log_file_path} 并启动 Dashboard...")
run_server(storage, host="127.0.0.1", port=8080)