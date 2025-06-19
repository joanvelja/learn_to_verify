import logging

import torch


# --- Log GPU Availability, Architecture, and Initial Memory ---
def gpu_info(initial_logger: logging.Logger):
    try:
        # Attempt to use pynvml for more detailed architecture info if available
        pynvml_available = False
        nvml_arch_map = {}
        try:
            import pynvml  # noqa: F401
            from pynvml.smi import nvidia_smi  # noqa: F401

            pynvml.nvmlInit()
            pynvml_available = True
            # Map NVML architecture enums to names (add more as needed based on pynvml constants)
            nvml_arch_map = {
                pynvml.NVML_DEVICE_ARCH_KEPLER: "Kepler",
                pynvml.NVML_DEVICE_ARCH_MAXWELL: "Maxwell",
                pynvml.NVML_DEVICE_ARCH_PASCAL: "Pascal",
                pynvml.NVML_DEVICE_ARCH_VOLTA: "Volta",
                pynvml.NVML_DEVICE_ARCH_TURING: "Turing",
                pynvml.NVML_DEVICE_ARCH_AMPERE: "Ampere",
                pynvml.NVML_DEVICE_ARCH_HOPPER: "Hopper",
                pynvml.NVML_DEVICE_ARCH_UNKNOWN: "Unknown",
            }
            initial_logger.info("Initialized pynvml for detailed GPU architecture reporting.")
        except ImportError:
            initial_logger.warning(
                "pynvml not found. Will fall back to using PyTorch compute capability for architecture estimation. Install nvidia-ml-py (`pip install nvidia-ml-py`) for more detailed info."
            )
        except pynvml.NVMLError as e:
            initial_logger.warning(f"Failed to initialize pynvml: {e}. Will fall back to PyTorch compute capability.")
            pynvml_available = False  # Ensure flag is false if init fails

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            initial_logger.info(f"Found {num_gpus} CUDA-enabled GPU(s) visible to PyTorch.")

            # Fallback compute capability map (used if pynvml fails or isn't installed)
            cc_arch_map = {
                3: "Kepler",
                5: "Maxwell",
                6: "Pascal",
                7: "Volta/Turing",  # Needs refinement based on minor version
                8: "Ampere",
                9: "Hopper",
                # Add future major versions here
            }

            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                total_memory_gb = props.total_memory / (1024**3)
                compute_capability = f"{props.major}.{props.minor}"
                arch_name = "Unknown"

                # 1. Try getting architecture from pynvml
                nvml_arch_success = False
                if pynvml_available:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        arch_enum = pynvml.nvmlDeviceGetArchitecture(handle)
                        arch_name = nvml_arch_map.get(arch_enum, f"Unknown NVML Arch ({arch_enum})")
                        nvml_arch_success = True
                    except pynvml.NVMLError as e:
                        initial_logger.warning(
                            f"pynvml failed to get architecture for GPU {i}: {e}. Falling back to compute capability."
                        )
                    except Exception as e:  # Catch potential unexpected errors
                        initial_logger.warning(
                            f"Unexpected error using pynvml for GPU {i}: {e}. Falling back to compute capability."
                        )

                # 2. Fallback to compute capability map if pynvml failed or wasn't available
                if not nvml_arch_success:
                    arch_name = cc_arch_map.get(props.major, "Unknown/Future CC")
                    # Refine Volta/Turing based on minor version if major is 7
                    if props.major == 7:
                        if props.minor == 0:
                            arch_name = "Volta"
                        elif props.minor == 5:
                            arch_name = "Turing"
                        else:
                            arch_name = f"Volta/Turing (CC {compute_capability})"  # Unknown minor version for 7.x

                # Get current memory stats for this device *for this process*
                allocated_memory_gb = 0.0
                reserved_memory_gb = 0.0
                memory_info_str = "Memory stats unavailable"
                try:
                    # Ensure context is on the correct device for memory query
                    with torch.cuda.device(i):
                        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                        reserved_memory_gb = torch.cuda.memory_reserved() / (
                            1024**3
                        )  # Includes allocated + cached by allocator
                    memory_info_str = f"Current Process Memory - Allocated: {allocated_memory_gb:.2f} GB, Reserved (Cached): {reserved_memory_gb:.2f} GB"
                except RuntimeError as e:
                    memory_info_str = f"Could not query memory stats: {e}"
                    initial_logger.warning(f"  Warning querying memory for GPU {i}: {e}")
                except Exception as e:  # Catch potential unexpected errors
                    memory_info_str = f"Unexpected error querying memory stats: {e}"
                    initial_logger.warning(f"  Unexpected error querying memory for GPU {i}: {e}")

                initial_logger.info(
                    f"  GPU {i}: {props.name}, Arch: {arch_name} (CC {compute_capability}), "
                    f"Total Memory: {total_memory_gb:.2f} GB. {memory_info_str}"
                )

            # Note about distributed training allocation
            initial_logger.info(
                "Note: The above list shows GPUs visible *before* framework allocation. "
                "Memory figures reflect the state *at this moment* for the current process. "
                "The distributed training framework (Accelerate/DeepSpeed) will manage final GPU allocation and memory usage based on its configuration and environment variables (e.g., CUDA_VISIBLE_DEVICES)."
            )

            # Shutdown pynvml if it was initialized
            if pynvml_available:
                try:
                    pynvml.nvmlShutdown()
                    initial_logger.info("Shut down pynvml.")
                except pynvml.NVMLError as e:
                    initial_logger.warning(f"Failed to shut down pynvml: {e}")

        else:
            initial_logger.warning(
                "torch.cuda.is_available() returned False. No CUDA-enabled GPUs detected by PyTorch. Training will likely proceed on CPU."
            )

    except ImportError:
        initial_logger.warning(
            "PyTorch ('torch') is not installed or could not be imported. Cannot check for GPU availability."
        )
    except Exception as e:
        initial_logger.error(
            f"An unexpected error occurred during GPU availability check: {e}",
            exc_info=True,
        )
        # Ensure pynvml is shut down if an error occurred after its init but before normal shutdown
        if "pynvml_available" in locals() and pynvml_available:
            try:
                pynvml.nvmlShutdown()
                initial_logger.info("Shut down pynvml after encountering an error during GPU check.")
            except NameError:  # pynvml might not be defined if import failed
                pass
            except pynvml.NVMLError as nvml_err:
                initial_logger.warning(f"Failed to shut down pynvml during error handling: {nvml_err}")
