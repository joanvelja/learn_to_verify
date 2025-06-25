import datetime
import json
import logging
import os
import platform
import signal
import subprocess
import threading
import time
import warnings
from pathlib import Path

import psutil

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger

# Optional imports - will use if available
try:
    import pynvml

    PYNVML_AVAILABLE = True
    logger.info("pynvml found. Enabling GPU monitoring.")

except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not found. Install with 'pip install nvidia-ml-py' for enhanced GPU monitoring.")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm  # noqa: F401

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class GPUMonitor:
    def __init__(
        self,
        log_dir="gpu_logs",
        log_file="gpu_monitor.log",
        json_log_file="gpu_data.json",
        check_interval_sec=1,
        memory_warning_threshold=90,  # percentage
        temp_warning_threshold=80,  # celsius
        utilization_warning_threshold=95,  # percentage
        log_to_console=True,
        log_to_file=True,
        capture_process_info=True,
        capture_memory_growth=True,
        memory_growth_window=10,  # Keep last N memory readings
        auto_start=False,
    ):
        # Initialize parameters
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file
        self.json_log_file = self.log_dir / json_log_file
        self.check_interval_sec = check_interval_sec
        self.memory_warning_threshold = memory_warning_threshold
        self.temp_warning_threshold = temp_warning_threshold
        self.utilization_warning_threshold = utilization_warning_threshold
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.capture_process_info = capture_process_info
        self.capture_memory_growth = capture_memory_growth
        self.memory_growth_window = memory_growth_window

        # Initialize state
        self.is_monitoring = False
        self.monitor_thread = None
        self.gpu_memory_history = {}  # {gpu_id: [mem1, mem2, ...]}
        self.gpu_count = 0
        self.iteration = 0
        self.start_time = None
        self.log_data = []

        # Set up logging
        self._setup_logging()

        # Initialize NVML if available
        if PYNVML_AVAILABLE:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(self.gpu_count):
                self.gpu_memory_history[i] = []

        # Start monitoring if requested
        if auto_start:
            self.start()

    def _setup_logging(self):
        """Set up logging directory and files"""
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)

            # Clear previous log files
            if os.path.exists(self.log_file):
                open(self.log_file, "w").close()
            if os.path.exists(self.json_log_file):
                open(self.json_log_file, "w").close()

    def start(self):
        """Start the GPU monitoring thread"""
        if self.is_monitoring:
            print("GPU monitoring is already running")
            return

        self.is_monitoring = True
        self.start_time = datetime.datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"GPU monitoring started. Logging every {self.check_interval_sec} seconds.")

        # Log system info
        self._log_system_info()

    def stop(self):
        """Stop the GPU monitoring thread"""
        if not self.is_monitoring:
            print("GPU monitoring is not running")
            return

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        # Save final JSON data
        if self.log_to_file and self.log_data:
            with open(self.json_log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)

        print("GPU monitoring stopped.")

    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        print("\nReceived termination signal. Shutting down GPU monitoring...")
        self.stop()

    def _get_gpu_info_nvml(self):
        """Get detailed GPU info using PYNVML"""
        if not PYNVML_AVAILABLE:
            return None

        gpu_info = []
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Basic GPU info
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

            # Memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory.total / 1024 / 1024  # Convert to MB
            memory_used = memory.used / 1024 / 1024
            memory_free = memory.free / 1024 / 1024
            memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0

            # Utilization info
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            memory_utilization = utilization.memory

            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                power_percent = (power_usage / power_limit) * 100 if power_limit > 0 else 0
            except pynvml.NVMLError:
                power_usage = power_limit = power_percent = 0

            # Clocks
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError:
                graphics_clock = sm_clock = mem_clock = 0

            # Process info
            processes = []
            if self.capture_process_info:
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in procs:
                        try:
                            process_info = {}
                            process_info["pid"] = p.pid
                            process_info["memory_used"] = p.usedGpuMemory / 1024 / 1024  # Convert to MB

                            # Get process name
                            try:
                                process = psutil.Process(p.pid)
                                process_info["name"] = process.name()
                                process_info["username"] = process.username()
                                process_info["cmdline"] = " ".join(process.cmdline())
                                process_info["cpu_percent"] = process.cpu_percent()
                                process_info["memory_percent"] = process.memory_percent()
                                process_info["create_time"] = datetime.datetime.fromtimestamp(
                                    process.create_time()
                                ).strftime("%Y-%m-%d %H:%M:%S")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                process_info["name"] = "Unknown"
                                process_info["username"] = "Unknown"
                                process_info["cmdline"] = "Unknown"

                            processes.append(process_info)
                        except Exception as e:
                            processes.append({"error": str(e)})
                except pynvml.NVMLError:
                    pass

            # Memory growth tracking
            if self.capture_memory_growth:
                self.gpu_memory_history[i].append(memory_used)
                if len(self.gpu_memory_history[i]) > self.memory_growth_window:
                    self.gpu_memory_history[i].pop(0)

                if len(self.gpu_memory_history[i]) >= 2:
                    memory_growth = self.gpu_memory_history[i][-1] - self.gpu_memory_history[i][0]
                    memory_growth_rate = memory_growth / len(self.gpu_memory_history[i])  # MB per interval
                else:
                    memory_growth = memory_growth_rate = 0
            else:
                memory_growth = memory_growth_rate = 0

            # Warning flags
            warnings = []
            if memory_percent > self.memory_warning_threshold:
                warnings.append(f"HIGH_MEMORY_USAGE: {memory_percent:.1f}%")
            if temperature > self.temp_warning_threshold:
                warnings.append(f"HIGH_TEMPERATURE: {temperature}°C")
            if gpu_utilization > self.utilization_warning_threshold:
                warnings.append(f"HIGH_UTILIZATION: {gpu_utilization}%")
            if memory_growth_rate > 5:  # Warning if growing by more than 5MB per interval
                warnings.append(f"MEMORY_LEAK_SUSPECT: {memory_growth_rate:.2f} MB/interval")

            gpu_info.append(
                {
                    "index": i,
                    "name": name,
                    "memory": {
                        "total": memory_total,
                        "used": memory_used,
                        "free": memory_free,
                        "percent": memory_percent,
                        "growth": memory_growth,
                        "growth_rate": memory_growth_rate,
                    },
                    "utilization": {
                        "gpu": gpu_utilization,
                        "memory": memory_utilization,
                    },
                    "temperature": temperature,
                    "power": {
                        "usage": power_usage,
                        "limit": power_limit,
                        "percent": power_percent,
                    },
                    "clocks": {
                        "graphics": graphics_clock,
                        "sm": sm_clock,
                        "memory": mem_clock,
                    },
                    "processes": processes,
                    "warnings": warnings,
                }
            )

        return gpu_info

    def _get_gpu_info_nvidia_smi(self):
        """Fallback to nvidia-smi if pynvml is not available"""
        try:
            nvidia_smi_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )

            gpu_info = []
            for line in nvidia_smi_output.strip().split("\n"):
                try:
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 8:
                        (
                            index,
                            name,
                            util,
                            mem_used,
                            mem_total,
                            temp,
                            power_draw,
                            power_limit,
                        ) = parts[:8]

                        # Convert values
                        index = int(index)
                        util = float(util)
                        mem_used = float(mem_used)
                        mem_total = float(mem_total)
                        mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        temp = float(temp)

                        try:
                            power_draw = float(power_draw)
                            power_limit = float(power_limit)
                            power_percent = (power_draw / power_limit) * 100 if power_limit > 0 else 0
                        except ValueError:
                            power_draw = power_limit = power_percent = 0

                        # Memory growth tracking
                        if self.capture_memory_growth:
                            if index not in self.gpu_memory_history:
                                self.gpu_memory_history[index] = []

                            self.gpu_memory_history[index].append(mem_used)
                            if len(self.gpu_memory_history[index]) > self.memory_growth_window:
                                self.gpu_memory_history[index].pop(0)

                            if len(self.gpu_memory_history[index]) >= 2:
                                memory_growth = self.gpu_memory_history[index][-1] - self.gpu_memory_history[index][0]
                                memory_growth_rate = memory_growth / len(self.gpu_memory_history[index])
                            else:
                                memory_growth = memory_growth_rate = 0
                        else:
                            memory_growth = memory_growth_rate = 0

                        # Warning flags
                        warnings = []
                        if mem_percent > self.memory_warning_threshold:
                            warnings.append(f"HIGH_MEMORY_USAGE: {mem_percent:.1f}%")
                        if temp > self.temp_warning_threshold:
                            warnings.append(f"HIGH_TEMPERATURE: {temp}°C")
                        if util > self.utilization_warning_threshold:
                            warnings.append(f"HIGH_UTILIZATION: {util}%")
                        if memory_growth_rate > 5:
                            warnings.append(f"MEMORY_LEAK_SUSPECT: {memory_growth_rate:.2f} MB/interval")

                        gpu_info.append(
                            {
                                "index": index,
                                "name": name,
                                "memory": {
                                    "total": mem_total,
                                    "used": mem_used,
                                    "free": mem_total - mem_used,
                                    "percent": mem_percent,
                                    "growth": memory_growth,
                                    "growth_rate": memory_growth_rate,
                                },
                                "utilization": {
                                    "gpu": util,
                                    "memory": 0,  # Not available in basic nvidia-smi
                                },
                                "temperature": temp,
                                "power": {
                                    "usage": power_draw,
                                    "limit": power_limit,
                                    "percent": power_percent,
                                },
                                "warnings": warnings,
                            }
                        )
                except (ValueError, IndexError):
                    pass

            return gpu_info
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error accessing nvidia-smi. Make sure NVIDIA drivers are installed.")
            return []

    def _get_pytorch_memory_info(self):
        """Get PyTorch-specific memory information if available"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        result = {}

        try:
            # Get number of devices
            device_count = torch.cuda.device_count()
            devices = {}

            for i in range(device_count):
                device_info = {
                    "allocated": torch.cuda.memory_allocated(i) / 1024 / 1024,  # MB
                    "reserved": torch.cuda.memory_reserved(i) / 1024 / 1024,  # MB
                    "cached": torch.cuda.memory_cached(i) / 1024 / 1024,  # MB
                }

                # Get memory stats if available (PyTorch 1.10+)
                try:
                    stats = torch.cuda.memory_stats(i)
                    device_info["stats"] = {
                        "num_alloc_retries": stats.get("num_alloc_retries", 0),
                        "num_ooms": stats.get("num_ooms", 0),
                        "max_split_size": stats.get("max_split_size", 0) / 1024 / 1024,  # MB
                        "segment_size": stats.get("segment_size", 0) / 1024 / 1024,  # MB
                    }
                except (AttributeError, RuntimeError):
                    pass

                devices[i] = device_info

            result["devices"] = devices
            result["max_memory_allocated"] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            result["max_memory_reserved"] = torch.cuda.max_memory_reserved() / 1024 / 1024  # MB

            # Get memory snapshot if available (PyTorch 1.10+)
            try:
                result["memory_snapshot"] = torch.cuda.memory_snapshot()
            except AttributeError:
                pass

        except Exception as e:
            result["error"] = str(e)

        return result

    def _log_system_info(self):
        """Log system information at startup"""
        system_info = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            },
            "libraries": {
                "pynvml": PYNVML_AVAILABLE,
                "torch": TORCH_AVAILABLE,
                "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
                "cuda_available": (torch.cuda.is_available() if TORCH_AVAILABLE else False),
                "cuda_version": (torch.version.cuda if TORCH_AVAILABLE and torch.cuda.is_available() else None),
            },
        }

        # Log to file
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                f.write("=== SYSTEM INFORMATION ===\n")
                json.dump(system_info, f, indent=2)
                f.write("\n\n")

        # Log to console
        if self.log_to_console:
            print("=== SYSTEM INFORMATION ===")
            print(json.dumps(system_info, indent=2))
            print()

    def _monitoring_loop(self):
        """Main monitoring loop that runs in a background thread"""
        while self.is_monitoring:
            try:
                # Collect GPU information
                if PYNVML_AVAILABLE:
                    gpu_info = self._get_gpu_info_nvml()
                else:
                    gpu_info = self._get_gpu_info_nvidia_smi()

                # Get PyTorch specific info if available
                pytorch_info = self._get_pytorch_memory_info() if TORCH_AVAILABLE else {}

                # Get system metrics
                system_metrics = {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                }

                # Combine all information
                timestamp = datetime.datetime.now()
                elapsed = (timestamp - self.start_time).total_seconds() if self.start_time else 0

                log_entry = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": elapsed,
                    "iteration": self.iteration,
                    "system": system_metrics,
                    "gpu": gpu_info,
                    "pytorch": pytorch_info,
                }

                # Store for later JSON export
                self.log_data.append(log_entry)

                # Log to file
                if self.log_to_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"\n=== GPU STATUS AT {timestamp} (ITERATION {self.iteration}) ===\n")

                        # System info
                        f.write(
                            f"System: CPU: {system_metrics['cpu_percent']}%, "
                            f"Memory: {system_metrics['memory_used_gb']:.2f}/{system_metrics['memory_total_gb']:.2f} GB "
                            f"({system_metrics['memory_percent']}%)\n"
                        )

                        # GPU info
                        if gpu_info:
                            for gpu in gpu_info:
                                f.write(f"\nGPU {gpu['index']} ({gpu['name']})\n")
                                f.write(
                                    f"  Memory: {gpu['memory']['used']:.2f}/{gpu['memory']['total']:.2f} MB "
                                    f"({gpu['memory']['percent']:.1f}%)\n"
                                )
                                f.write(
                                    f"  Utilization: {gpu['utilization']['gpu']}%, "
                                    f"Temperature: {gpu['temperature']}°C, "
                                    f"Power: {gpu['power']['usage']:.2f}/{gpu['power']['limit']:.2f}W\n"
                                )

                                if "clocks" in gpu:
                                    f.write(
                                        f"  Clocks - Graphics: {gpu['clocks']['graphics']} MHz, "
                                        f"Memory: {gpu['clocks']['memory']} MHz\n"
                                    )

                                if gpu["memory"]["growth_rate"] != 0:
                                    f.write(
                                        f"  Memory growth: {gpu['memory']['growth_rate']:.2f} MB/interval "
                                        f"(total change: {gpu['memory']['growth']:.2f} MB)\n"
                                    )

                                if gpu["warnings"]:
                                    f.write(f"  WARNINGS: {', '.join(gpu['warnings'])}\n")

                                if gpu.get("processes"):
                                    f.write(f"  Active processes ({len(gpu['processes'])}):\n")
                                    for proc in gpu["processes"]:
                                        f.write(
                                            f"    PID {proc['pid']}: {proc['name']} ({proc['memory_used']:.2f} MB)\n"
                                        )
                                        f.write(f"        CMD: {proc.get('cmdline', 'Unknown')}\n")
                        else:
                            f.write("No GPU information available\n")

                        # PyTorch info
                        if pytorch_info:
                            f.write("\nPyTorch CUDA Memory:\n")
                            f.write(f"  Max Allocated: {pytorch_info.get('max_memory_allocated', 0):.2f} MB\n")
                            f.write(f"  Max Reserved: {pytorch_info.get('max_memory_reserved', 0):.2f} MB\n")

                            if "devices" in pytorch_info:
                                for device_id, device_info in pytorch_info["devices"].items():
                                    f.write(
                                        f"  Device {device_id}: "
                                        f"Allocated: {device_info['allocated']:.2f} MB, "
                                        f"Reserved: {device_info['reserved']:.2f} MB\n"
                                    )

                # Log to console
                if self.log_to_console:
                    print(f"\n=== GPU STATUS AT {timestamp} (ITERATION {self.iteration}) ===")

                    # System info
                    print(
                        f"System: CPU: {system_metrics['cpu_percent']}%, "
                        f"Memory: {system_metrics['memory_used_gb']:.2f}/{system_metrics['memory_total_gb']:.2f} GB "
                        f"({system_metrics['memory_percent']}%)"
                    )

                    # GPU info
                    if gpu_info:
                        for gpu in gpu_info:
                            mem_percent = gpu["memory"]["percent"]
                            temp = gpu["temperature"]
                            util = gpu["utilization"]["gpu"]

                            # Visual indicators for critical metrics
                            mem_indicator = self._get_visual_indicator(mem_percent, self.memory_warning_threshold)
                            temp_indicator = self._get_visual_indicator(temp, self.temp_warning_threshold)
                            util_indicator = self._get_visual_indicator(util, self.utilization_warning_threshold)

                            print(f"\nGPU {gpu['index']} ({gpu['name']})")
                            print(
                                f"  Memory: {gpu['memory']['used']:.2f}/{gpu['memory']['total']:.2f} MB "
                                f"({mem_percent:.1f}%) {mem_indicator}"
                            )
                            print(
                                f"  Utilization: {util}% {util_indicator}, "
                                f"Temperature: {temp}°C {temp_indicator}, "
                                f"Power: {gpu['power']['usage']:.2f}/{gpu['power']['limit']:.2f}W"
                            )

                            if "clocks" in gpu:
                                print(
                                    f"  Clocks - Graphics: {gpu['clocks']['graphics']} MHz, "
                                    f"Memory: {gpu['clocks']['memory']} MHz"
                                )

                            if gpu["memory"]["growth_rate"] != 0:
                                growth_indicator = "↗" if gpu["memory"]["growth_rate"] > 0 else "↘"
                                print(
                                    f"  Memory growth: {gpu['memory']['growth_rate']:.2f} MB/interval {growth_indicator} "
                                    f"(total change: {gpu['memory']['growth']:.2f} MB)"
                                )

                            if gpu["warnings"]:
                                print(f"  ⚠️ WARNINGS: {', '.join(gpu['warnings'])}")

                            if gpu.get("processes"):
                                print(f"  Active processes ({len(gpu['processes'])}):")
                                for proc in gpu["processes"]:
                                    print(f"    PID {proc['pid']}: {proc['name']} ({proc['memory_used']:.2f} MB)")
                                    if self.capture_process_info:
                                        print(f"        CMD: {proc.get('cmdline', 'Unknown')}")
                    else:
                        print("No GPU information available")

                    # PyTorch info
                    if pytorch_info and pytorch_info.get("devices"):
                        print("\nPyTorch CUDA Memory:")
                        print(f"  Max Allocated: {pytorch_info.get('max_memory_allocated', 0):.2f} MB")
                        print(f"  Max Reserved: {pytorch_info.get('max_memory_reserved', 0):.2f} MB")

                        for device_id, device_info in pytorch_info["devices"].items():
                            print(
                                f"  Device {device_id}: "
                                f"Allocated: {device_info['allocated']:.2f} MB, "
                                f"Reserved: {device_info['reserved']:.2f} MB"
                            )

                # Increment iteration
                self.iteration += 1

                # Sleep until next check
                time.sleep(self.check_interval_sec)

            except Exception as e:
                print(f"Error in GPU monitoring: {str(e)}")
                time.sleep(self.check_interval_sec)

    def _get_visual_indicator(self, value, threshold):
        """Get visual indicator based on value and threshold"""
        if value >= threshold:
            return "🔴"  # Red circle for warning
        elif value >= threshold * 0.8:
            return "🟠"  # Orange circle for caution
        elif value >= threshold * 0.6:
            return "🟡"  # Yellow circle for moderate
        else:
            return "🟢"  # Green circle for good

    def log_iteration(self, iteration=None, custom_info=None):
        """Log GPU status at specific iteration with optional custom info"""
        if iteration is not None:
            self.iteration = iteration

        if not self.is_monitoring:
            # Get GPU info directly if not in monitoring mode
            if PYNVML_AVAILABLE:
                gpu_info = self._get_gpu_info_nvml()
            else:
                gpu_info = self._get_gpu_info_nvidia_smi()

            # Format and log the information
            timestamp = datetime.datetime.now()
            log_entry = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "iteration": self.iteration,
                "gpu": gpu_info,
                "custom": custom_info,
            }

            # Store for later JSON export
            self.log_data.append(log_entry)

            # Log to file and console as needed
            # Similar to the monitoring loop code...
        else:
            # Just update the iteration number if monitoring is active
            self.iteration = iteration

            # Add custom info
            if custom_info:
                self.log_data[-1]["custom"] = custom_info

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.stop()

    def __del__(self):
        """Cleanup on object destruction"""
        self.stop()
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
