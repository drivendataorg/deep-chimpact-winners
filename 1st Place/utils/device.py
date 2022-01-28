import tensorflow as tf
import psutil


def get_device():
    gpus = tf.config.list_logical_devices('GPU')
    ngpu = len(gpus)
    if ngpu: # if number of GPUs are 0 then CPU
        strategy = tf.distribute.MirroredStrategy(gpus) # single-GPU or multi-GPU
        print("> Running on GPU", end=' | ')
        print("Num of GPUs: ", ngpu)
        device='GPU'
    else:
        print("> Running on CPU")
        strategy = tf.distribute.get_strategy()
        device='CPU'
    return strategy, device

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
        
def get_config():
    print("="*10, "Device Config", "="*10)
    print(f'CPU Usage: {psutil.cpu_percent()}%')
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}") ; print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}") ; print(f"Percentage: {svmem.percent}%")
    return