import csv
import io
import json
import platform
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return " ".join(winreg.QueryValueEx(key, "ProcessorNameString")[0].split())
        except (OSError, ImportError):
            pass
    name = platform.processor() or platform.uname().processor
    return " ".join(name.split()) if name else "Unknown CPU"


def nvidia_smi_stats():
    fields = ["index", "name", "utilization.gpu", "memory.total", "memory.used", "temperature.gpu"]
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != len(fields):
                continue
            total_mib, used_mib = float(row[3]), float(row[4])
            gpus.append({
                "id": int(row[0]), "name": row[1].strip(), "gpu_util": float(row[2]),
                "mem_total": int(total_mib * 1024 * 1024), "mem_used": int(used_mib * 1024 * 1024),
                "mem_free": int((total_mib - used_mib) * 1024 * 1024),
                "temperature": float(row[5]), "source": "nvidia-smi",
            })
        return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def gpu_stats():
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            result = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temperature = 0
                result.append({
                    "id": index, "name": name, "gpu_util": util.gpu,
                    "mem_total": memory.total, "mem_used": memory.used, "mem_free": memory.free,
                    "temperature": temperature, "source": "nvml",
                })
            if result:
                return result
        except Exception:
            pass
    return nvidia_smi_stats()


def disk_stats():
    disks = []
    for partition in psutil.disk_partitions(all=False):
        try:
            if "cdrom" in partition.opts.lower():
                continue
            usage = psutil.disk_usage(partition.mountpoint)
            disks.append({"device": partition.device, "mountpoint": partition.mountpoint,
                          "total": usage.total, "used": usage.used, "free": usage.free,
                          "percent": usage.percent})
        except (PermissionError, OSError):
            continue
    return disks


def main():
    if psutil is None:
        print("psutil is required for CPU and memory metrics", file=sys.stderr, flush=True)
        return 2
    model = cpu_name()
    psutil.cpu_percent(interval=None)
    while True:
        try:
            memory = psutil.virtual_memory()
            stats = {
                "cpu_model": model, "cpu_percent": psutil.cpu_percent(interval=0.15),
                "memory": {"total": memory.total, "available": memory.available,
                           "percent": memory.percent, "used": memory.used},
                "disks": disk_stats(), "gpus": gpu_stats(), "timestamp": time.time(),
            }
            print(f"__JSON_START__{json.dumps(stats, ensure_ascii=False)}__JSON_END__", flush=True)
            time.sleep(0.85)
        except KeyboardInterrupt:
            return 0
        except Exception as error:
            print(f"__JSON_START__{json.dumps({'error': str(error)})}__JSON_END__", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
