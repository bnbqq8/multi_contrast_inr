# # test all subjects in test set
# import json
# import os
# from pathlib import Path

# # python3 main.py --logging --config config/config_brats_mlpv2_custom.yaml
# with open("../IXI_preprocess/dataset_split_20251225.json") as f:
#     datalist = json.load(f)
# patients = [i.split("/")[-1] for i in datalist["test"]]
# for patient in patients:
#     cmd = f"python3 main.py --logging --config config/config_brats_mlpv2_custom.yaml --subject_id {patient}"
#     os.system(cmd)
import json
import multiprocessing as mp
import os
from pathlib import Path


def worker(gpu_id, task_queue, config_path):
    """
    每个 GPU 进程执行的函数
    """
    while True:
        # 从队列中获取任务
        patient = task_queue.get()
        if patient is None:  # 收到终止信号
            break

        print(f"进程 [GPU {gpu_id}] 开始处理病人: {patient}")

        # 构造命令，通过指定 CUDA_VISIBLE_DEVICES 锁定显卡
        # 注意：这里调用 main.py 时显卡编号应设为 0，因为环境变量已经做了隔离
        cmd = (
            f"python3 main.py "
            f"--logging "
            f"--config {config_path} "
            f"--subject_id {patient} "
            f"--cuda_visible_device {gpu_id}"
        )

        exit_code = os.system(cmd)

        if exit_code != 0:
            print(f"错误: 病人 {patient} 在 GPU {gpu_id} 上执行失败。")

        task_queue.task_done()


if __name__ == "__main__":
    # 1. 配置参数
    CONFIG = "config/config_brats_mlpv2_custom.yaml"
    JSON_PATH = "../IXI_preprocess/dataset_split_20251225.json"
    GPUS = [0, 1, 2]  # 使用 0, 1, 2 号卡

    # 2. 读取病人列表
    with open(JSON_PATH) as f:
        datalist = json.load(f)
    patients = [i.split("/")[-1] for i in datalist["test"]]

    # 3. 创建任务队列
    task_queue = mp.JoinableQueue()
    for patient in patients:
        task_queue.put(patient)

    # 4. 启动并发进程
    processes = []
    for gpu_id in GPUS:
        # 为每张显卡开启一个守护进程
        p = mp.Process(target=worker, args=(gpu_id, task_queue, CONFIG))
        p.daemon = True
        p.start()
        processes.append(p)

    # 5. 等待所有任务完成
    task_queue.join()

    # 6. 停止所有 worker
    for _ in GPUS:
        task_queue.put(None)
    for p in processes:
        p.join()

    print("所有测试任务已完成。")
