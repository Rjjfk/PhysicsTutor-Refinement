import os
import json
import subprocess
import psutil
import time
import signal
import random
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # 解决跨域问题

# 全局状态管理
training_state = {
    "status": "idle",  # idle, running, paused, completed, stopped, failed
    "process": None,
    "pid": None,
    "start_time": None,
    "elapsed_time": "00:00:00",
    "epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "best_loss": float('inf'),
    "metrics": {
        "physics_accuracy": 0,
        "role_accuracy": 0,
        "stability": 0,
        "data_usage": 0
    },
    "epoch_history": [],
    "config": {}
}

# 日志管理
log_entries = []
MAX_LOG_ENTRIES = 1000

# 训练配置默认值
DEFAULT_CONFIG = {
    "task_type": "physics_adv",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "learning_rate": 1e-4,
    "data_path": "./data/processed/physics_train",
    "output_dir": "./model/physics_adversarial_lora",
    "save_interval": 2,
    "auto_save": True,
    "shuffle_data": True,
    "augment_data": False,
    "base_model_path": "./model/physics_lora",
    "loop_count": 2,
    "generate_num_samples": 500,
    "physics_model_path": "",
    "role_model_path": ""
}


def add_log(message, level="info"):
    """添加日志条目"""
    global log_entries
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "level": level
    }
    log_entries.append(log_entry)
    # 限制日志数量
    if len(log_entries) > MAX_LOG_ENTRIES:
        log_entries = log_entries[-MAX_LOG_ENTRIES:]
    print(f"[{timestamp}] {message}")


def get_gpu_info():
    """获取GPU信息"""
    gpu_info = {
        "available": False,
        "name": "N/A",
        "memory": "N/A",
        "utilization": "N/A"
    }

    try:
        # 使用nvidia-smi命令获取GPU信息
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )

        if result.strip():
            parts = result.strip().split(",")
            if len(parts) >= 4:
                gpu_info["available"] = True
                gpu_info["name"] = parts[0].strip()
                gpu_info["memory"] = f"{parts[1].strip()} MiB"
                gpu_info["utilization"] = f"{parts[3].strip()}%"
    except (subprocess.SubprocessError, FileNotFoundError):
        # 没有NVIDIA GPU或nvidia-smi不可用
        pass

    return gpu_info


def get_system_info():
    """获取系统硬件信息"""
    # CPU信息
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=0.1)

    # 内存信息
    mem = psutil.virtual_memory()
    mem_total = f"{mem.total / (1024 ** 3):.2f} GB"
    mem_available = f"{mem.available / (1024 ** 3):.2f} GB"
    mem_usage = f"{mem.percent}%"

    return {
        "gpu": get_gpu_info(),
        "cpu": {
            "name": get_cpu_name(),
            "cores": cpu_count,
            "threads": cpu_threads,
            "usage": cpu_usage
        },
        "memory": {
            "total": mem_total,
            "available": mem_available,
            "usage": mem_usage
        }
    }


def get_cpu_name():
    """获取CPU名称"""
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        # Windows系统
        import wmi
        c = wmi.WMI()
        for processor in c.Win32_Processor():
            return processor.Name
    except:
        return "Unknown CPU"


def update_training_time():
    """更新训练 elapsed time"""
    if training_state["start_time"] and training_state["status"] in ["running", "paused"]:
        elapsed = datetime.now() - training_state["start_time"]
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        training_state["elapsed_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"


def simulate_training(config):
    """模拟训练过程（实际应用中替换为真实训练代码）"""
    global training_state

    try:
        training_state["status"] = "running"
        training_state["start_time"] = datetime.now()
        training_state["total_epochs"] = config["num_train_epochs"]
        training_state["best_loss"] = float('inf')
        training_state["epoch_history"] = []

        add_log(f"开始{get_task_type_name(config['task_type'])}，总轮次: {config['num_train_epochs']}", "info")

        # 模拟多轮训练
        for epoch in range(1, config["num_train_epochs"] + 1):
            # 检查是否需要停止
            if training_state["status"] == "stopped":
                add_log("训练被手动终止", "info")
                break

            # 检查是否需要暂停
            while training_state["status"] == "paused":
                time.sleep(1)
                update_training_time()

            if training_state["status"] == "stopped":
                add_log("训练被手动终止", "info")
                break

            training_state["epoch"] = epoch
            add_log(f"开始第 {epoch}/{config['num_train_epochs']} 轮训练", "info")

            # 模拟训练过程（每轮持续5-10秒）
            epoch_start = time.time()
            current_loss = 0.5 - (epoch * 0.05) + random.uniform(-0.03, 0.03)
            current_loss = max(0.01, current_loss)  # 确保损失为正

            # 更新指标（模拟）
            physics_acc = min(100, 40 + epoch * 8 + random.randint(-5, 5))
            role_acc = min(100, 35 + epoch * 7 + random.randint(-5, 5))
            stability = min(100, 50 + epoch * 6 + random.randint(-3, 3))
            data_usage = min(100, 30 + epoch * 10 + random.randint(-2, 2))

            # 模拟训练耗时
            for i in range(10):
                if training_state["status"] in ["paused", "stopped"]:
                    break
                time.sleep(random.uniform(0.5, 1.0))  # 每步耗时
                update_training_time()

                # 实时更新损失（模拟训练过程）
                step_loss = current_loss + random.uniform(-0.02, 0.02)
                training_state["current_loss"] = step_loss
                training_state["metrics"] = {
                    "physics_accuracy": physics_acc * (i + 1) / 10,
                    "role_accuracy": role_acc * (i + 1) / 10,
                    "stability": stability * (i + 1) / 10,
                    "data_usage": data_usage * (i + 1) / 10
                }

            if training_state["status"] in ["paused", "stopped"]:
                break

            # 完成本轮训练
            epoch_time = time.time() - epoch_start
            minutes, seconds = divmod(int(epoch_time), 60)
            epoch_time_str = f"{minutes:02}:{seconds:02}"

            # 更新状态
            training_state["current_loss"] = current_loss
            if current_loss < training_state["best_loss"]:
                training_state["best_loss"] = current_loss

            training_state["metrics"] = {
                "physics_accuracy": physics_acc,
                "role_accuracy": role_acc,
                "stability": stability,
                "data_usage": data_usage
            }

            # 记录历史
            training_state["epoch_history"].append({
                "epoch": epoch,
                "loss": current_loss,
                "physics_accuracy": physics_acc,
                "role_accuracy": role_acc,
                "time": epoch_time_str
            })

            add_log(
                f"第 {epoch} 轮训练完成，损失: {current_loss:.4f}, 物理准确率: {physics_acc}%, 耗时: {epoch_time_str}",
                "info")

            # 自动保存
            if config["auto_save"] and (epoch % config["save_interval"] == 0 or epoch == config["num_train_epochs"]):
                save_result = save_current_model(epoch, config["output_dir"])
                if save_result["success"]:
                    add_log(f"自动保存模型（轮次 {epoch}）到 {config['output_dir']}", "success")
                else:
                    add_log(f"自动保存模型失败: {save_result['message']}", "error")

        # 训练完成
        if training_state["status"] != "stopped" and training_state["status"] != "failed":
            training_state["status"] = "completed"
            add_log(f"所有{config['num_train_epochs']}轮训练已完成！", "success")
    except Exception as e:
        training_state["status"] = "failed"
        add_log(f"训练过程出错: {str(e)}", "error")
    finally:
        training_state["process"] = None
        training_state["pid"] = None
        update_training_time()


def get_task_type_name(task_type):
    """获取任务类型的中文名称"""
    task_names = {
        "physics_adv": "物理对抗训练",
        "role_adv": "角色对抗训练",
        "physics_self": "物理自训练",
        "role_self": "角色自训练",
        "merge": "合并微调"
    }
    return task_names.get(task_type, "未知训练任务")


def save_current_model(epoch, save_path):
    """保存当前模型（实际应用中替换为真实保存逻辑）"""
    try:
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

        # 模拟模型保存
        model_info = {
            "epoch": epoch,
            "loss": training_state["current_loss"],
            "best_loss": training_state["best_loss"],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": training_state["metrics"]
        }

        # 保存模型信息到JSON文件
        with open(os.path.join(save_path, f"model_epoch_{epoch}.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # 保存最终模型链接
        if os.path.exists(os.path.join(save_path, "final_model.json")):
            os.remove(os.path.join(save_path, "final_model.json"))
        os.symlink(f"model_epoch_{epoch}.json", os.path.join(save_path, "final_model.json"))

        return {
            "success": True,
            "message": f"模型已保存到 {save_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"保存模型失败: {str(e)}"
        }


@app.route('/api/system-info', methods=['GET'])
def system_info():
    """获取系统硬件信息"""
    try:
        info = get_system_info()
        return jsonify({
            "success": True,
            "data": info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取系统信息失败: {str(e)}"
        })


@app.route('/api/start-training', methods=['POST'])
def start_training():
    """启动训练"""
    global training_state

    if training_state["status"] in ["running", "paused"]:
        return jsonify({
            "success": False,
            "message": "已有训练任务在运行中，请先停止当前任务"
        })

    try:
        # 获取并验证配置
        config = request.json or {}
        # 合并默认配置
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value

        # 验证必要参数
        if not config["data_path"]:
            return jsonify({
                "success": False,
                "message": "训练数据集路径不能为空"
            })

        if config["task_type"] == "merge" and (not config["physics_model_path"] or not config["role_model_path"]):
            return jsonify({
                "success": False,
                "message": "合并微调需要指定物理模型和角色模型路径"
            })

        # 保存配置
        training_state["config"] = config

        # 启动训练线程（实际应用中可以替换为subprocess调用真实训练脚本）
        training_thread = threading.Thread(target=simulate_training, args=(config,), daemon=True)
        training_thread.start()

        return jsonify({
            "success": True,
            "message": f"{get_task_type_name(config['task_type'])}已启动",
            "data": {
                "task_type": config["task_type"],
                "total_epochs": config["num_train_epochs"]
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"启动训练失败: {str(e)}"
        })


@app.route('/api/pause-training', methods=['POST'])
def pause_training():
    """暂停训练"""
    global training_state

    if training_state["status"] != "running":
        return jsonify({
            "success": False,
            "message": "当前没有正在运行的训练任务"
        })

    try:
        training_state["status"] = "paused"
        add_log("训练已暂停", "info")
        return jsonify({
            "success": True,
            "message": "训练已暂停"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"暂停训练失败: {str(e)}"
        })


@app.route('/api/resume-training', methods=['POST'])
def resume_training():
    """恢复训练"""
    global training_state

    if training_state["status"] != "paused":
        return jsonify({
            "success": False,
            "message": "当前训练任务未处于暂停状态"
        })

    try:
        training_state["status"] = "running"
        add_log("训练已恢复", "info")
        return jsonify({
            "success": True,
            "message": "训练已恢复"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"恢复训练失败: {str(e)}"
        })


@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    """停止训练"""
    global training_state

    if training_state["status"] not in ["running", "paused"]:
        return jsonify({
            "success": False,
            "message": "当前没有正在运行的训练任务"
        })

    try:
        training_state["status"] = "stopped"

        # 如果有子进程，终止它
        if training_state["process"] and training_state["process"].poll() is None:
            try:
                # 终止子进程及其所有子进程
                parent = psutil.Process(training_state["pid"])
                for child in parent.children(recursive=True):
                    child.send_signal(signal.SIGTERM)
                training_state["process"].send_signal(signal.SIGTERM)
                time.sleep(1)
                if training_state["process"].poll() is None:
                    training_state["process"].send_signal(signal.SIGKILL)
            except Exception as e:
                add_log(f"终止训练进程时出错: {str(e)}", "warning")

        add_log("训练已终止", "info")
        return jsonify({
            "success": True,
            "message": "训练已终止"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"停止训练失败: {str(e)}"
        })


@app.route('/api/save-model', methods=['POST'])
def save_model():
    """保存当前模型"""
    if training_state["status"] not in ["running", "paused"]:
        return jsonify({
            "success": False,
            "message": "当前没有正在运行的训练任务"
        })

    try:
        data = request.json or {}
        epoch = data.get("epoch", training_state["epoch"])
        save_path = data.get("save_path", training_state["config"].get("output_dir", "./model"))

        result = save_current_model(epoch, save_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"保存模型失败: {str(e)}"
        })


@app.route('/api/training-status', methods=['GET'])
def training_status():
    """获取训练状态"""
    update_training_time()
    return jsonify({
        "success": True,
        "data": {
            "status": training_state["status"],
            "epoch": training_state["epoch"],
            "total_epochs": training_state["total_epochs"],
            "elapsed_time": training_state["elapsed_time"],
            "current_loss": training_state["current_loss"],
            "best_loss": training_state["best_loss"],
            "metrics": training_state["metrics"],
            "epoch_history": training_state["epoch_history"],
            "config": training_state["config"]
        }
    })


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """获取日志"""
    return jsonify({
        "success": True,
        "data": log_entries
    })


@app.route('/api/clear-logs', methods=['POST'])
def clear_logs():
    """清空日志"""
    global log_entries
    log_entries = []
    add_log("日志已清空", "info")
    return jsonify({
        "success": True,
        "message": "日志已清空"
    })


@app.route('/api/config-defaults', methods=['GET'])
def get_config_defaults():
    """获取默认配置"""
    task_type = request.args.get("task_type", "physics_adv")
    # 根据任务类型返回对应默认配置
    config = DEFAULT_CONFIG.copy()

    # 针对不同任务类型的特定默认值
    task_specific = {
        "physics_adv": {
            "data_path": "./data/processed/physics_train",
            "output_dir": "./model/physics_adversarial_lora",
            "num_train_epochs": 3,
        },
        "role_adv": {
            "data_path": "./data/processed/elysia_train",
            "output_dir": "./model/role_adversarial_lora",
            "num_train_epochs": 2,
        },
        "physics_self": {
            "data_path": "./data/processed/physics_train",
            "output_dir": "./model/physics_self_lora",
            "num_train_epochs": 2,
            "learning_rate": 5e-5,
        },
        "role_self": {
            "data_path": "./data/processed/elysia_train",
            "output_dir": "./model/role_self_lora",
            "num_train_epochs": 2,
            "learning_rate": 5e-5,
        },
        "merge": {
            "data_path": "./data/processed/combined_train",
            "output_dir": "./model/merged_final_model",
            "num_train_epochs": 1,
            "learning_rate": 3e-5,
            "physics_model_path": "./model/physics_self_lora/final_model",
            "role_model_path": "./model/role_self_lora/final_model",
        }
    }

    if task_type in task_specific:
        config.update(task_specific[task_type])

    return jsonify({
        "success": True,
        "data": config
    })


if __name__ == '__main__':
    add_log("训练系统后端已启动", "info")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
