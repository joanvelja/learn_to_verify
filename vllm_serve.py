import argparse
import logging
import os
from dataclasses import dataclass, field
from collections.abc import Sequence

import torch

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_available,
)


if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI, HTTPException


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
else:
    Worker = object

logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorker(Worker):
    """
    A vLLM worker that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    def __init__(self, *args, **kwargs):
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )

        super().__init__(*args, **kwargs)

        # The following attributes are initialized when `init_communicator` method is called.
        self.pynccl_comm = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. Call close_communicator first."
            )

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(
        self, name: str, dtype: torch.dtype, shape: Sequence[int]
    ) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(
            weight, src=self.client_rank, stream=torch.cuda.current_stream()
        )
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


@dataclass
class ModelConfig:
    """Configuration for a single model in the multi-model server."""

    model_path: str
    model_id: str  # A unique identifier for the model
    revision: str | None = None
    max_model_len: int | None = None
    enable_prefix_caching: bool | None = None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the multi-model vLLM serve script.

    Args:
        models (`str`):
            Comma-separated list of model paths and IDs in format "path1:id1,path2:id2".
        revisions (`str` or `None`, *optional*, defaults to `None`):
            Comma-separated list of revisions for each model. Use empty string for default revision.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use for each model.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`list[float]`, *optional*, defaults to `[0.9]`):
            List of ratios (between 0 and 1) *must sum to ~1* of GPU memory to reserve for the model weights, activations, and KV cache.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation.
        max_model_lens (`str` or `None`, *optional*, defaults to `None`):
            Comma-separated list of max_model_len values for each model. Use empty string for default.
        enable_prefix_cachings (`str` or `None`, *optional*, defaults to `None`):
            Comma-separated list of boolean values (True/False) for enabling prefix caching for each model.
    """

    models: str = field(
        metadata={
            "help": "Comma-separated list of model paths and IDs in format 'path1:id1,path2:id2'."
        }
    )
    revisions: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated list of revisions for each model. Use empty string for default revision."
        },
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use for each model."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: list[float] = field(
        default=[0.9],
        metadata={
            "help": "List of ratios (between 0 and 1) *must sum to ~1* of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration."
        },
    )
    max_model_lens: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated list of max_model_len values for each model. Use empty string for default."
        },
    )
    enable_prefix_cachings: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated list of boolean values (True/False) for enabling prefix caching for each model."
        },
    )

    def parse_model_configs(self) -> list[ModelConfig]:
        """Parse model configurations from the command line arguments."""
        model_configs = []

        # Parse model paths and IDs
        model_entries = self.models.split(",")
        model_paths_ids = [entry.split(":") for entry in model_entries]
        if any(len(entry) != 2 for entry in model_paths_ids):
            raise ValueError("Model entries must be in the format 'path:id'")

        # Parse revisions if provided
        revisions = [""] * len(model_paths_ids)
        if self.revisions:
            revisions_list = self.revisions.split(",")
            if len(revisions_list) != len(model_paths_ids):
                raise ValueError("Number of revisions must match number of models")
            revisions = [rev if rev else None for rev in revisions_list]

        # Parse max_model_lens if provided
        max_model_lens = [None] * len(model_paths_ids)
        if self.max_model_lens:
            max_model_lens_list = self.max_model_lens.split(",")
            if len(max_model_lens_list) != len(model_paths_ids):
                raise ValueError(
                    "Number of max_model_len values must match number of models"
                )
            max_model_lens = [
                int(max_len) if max_len else None for max_len in max_model_lens_list
            ]

        # Parse enable_prefix_cachings if provided
        enable_prefix_cachings = [None] * len(model_paths_ids)
        if self.enable_prefix_cachings:
            enable_prefix_cachings_list = self.enable_prefix_cachings.split(",")
            if len(enable_prefix_cachings_list) != len(model_paths_ids):
                raise ValueError(
                    "Number of enable_prefix_caching values must match number of models"
                )
            enable_prefix_cachings = [
                (caching.lower() == "true") if caching else None
                for caching in enable_prefix_cachings_list
            ]

        # Check that gpu_memory_utilization sums to 1
        if sum(self.gpu_memory_utilization) > 1.0:
            raise ValueError("gpu_memory_utilization sum must be <= 1.0")

        # Create ModelConfig objects
        for i, (path_id, revision, max_model_len, enable_prefix_caching) in enumerate(
            zip(model_paths_ids, revisions, max_model_lens, enable_prefix_cachings)
        ):
            path, model_id = path_id
            model_configs.append(
                ModelConfig(
                    model_path=path,
                    model_id=model_id,
                    revision=revision,
                    max_model_len=max_model_len,
                    enable_prefix_caching=enable_prefix_caching,
                )
            )

        return model_configs


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError(
            "vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`."
        )

    # Parse model configurations
    model_configs = script_args.parse_model_configs()

    # Initialize a dictionary to store LLM instances
    llm_instances: dict[str, LLM] = {}

    # Load all models
    for config in model_configs:
        logger.info(f"Loading model {config.model_id} from {config.model_path}")
        llm = LLM(
            model=config.model_path,
            revision=config.revision,
            tensor_parallel_size=script_args.tensor_parallel_size,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            dtype=script_args.dtype,
            enable_prefix_caching=config.enable_prefix_caching,
            max_model_len=config.max_model_len,
            worker_cls=WeightSyncWorker,
        )
        llm_instances[config.model_id] = llm

    # Get list of available model IDs
    available_models = list(llm_instances.keys())
    logger.info(f"Loaded {len(available_models)} models: {', '.join(available_models)}")

    app = FastAPI()

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok", "available_models": available_models}

    @app.get("/models/")
    async def list_models():
        """
        List all available models on the server.
        """
        return {"models": available_models}

    @app.get("/get_tensor_parallel_size/{model_id}")
    async def get_tensor_parallel_size(model_id: str):
        """
        Retrieves the tensor parallel size from the specified model's LLM engine.

        Args:
            model_id (`str`): The ID of the model to query.

        Returns:
            `dict`: A dictionary containing the tensor parallel size.

        Example response:
        ```json
        {"tensor_parallel_size": 8}
        ```
        """
        if model_id not in llm_instances:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "tensor_parallel_size": llm_instances[
                model_id
            ].llm_engine.parallel_config.tensor_parallel_size
        }

    class GenerateRequest(BaseModel):
        model_id: str
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: str | None = None

    class GenerateResponse(BaseModel):
        model_id: str
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts using the specified model.

        Args:
            request (`GenerateRequest`):
                - `model_id` (`str`): The ID of the model to use for generation.
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `model_id` (`str`): The ID of the model used for generation.
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"model_id": "model1", "prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"model_id": "model1", "completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """
        if request.model_id not in llm_instances:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model_id} not found"
            )

        # Get the requested model
        llm = llm_instances[request.model_id]

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        all_outputs = llm.generate(request.prompts, sampling_params=sampling_params)
        completion_ids = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"model_id": request.model_id, "completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        model_id: str
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(
        request: InitCommunicatorRequest, background_tasks: BackgroundTasks
    ):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers for the specified model.

        Args:
            request (`InitCommunicatorRequest`):
                - `model_id` (`str`): The ID of the model to initialize the communicator for.
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        if request.model_id not in llm_instances:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model_id} not found"
            )

        # Get the requested model
        llm = llm_instances[request.model_id]

        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(request.host, request.port, script_args.tensor_parallel_size + 1),
        )
        return {
            "message": f"Request received, initializing communicator for model {request.model_id}"
        }

    class UpdateWeightsRequest(BaseModel):
        model_id: str
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(
        request: UpdateWeightsRequest, background_tasks: BackgroundTasks
    ):
        """
        Updates the weights of the specified model with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `model_id` (`str`): The ID of the model to update the weights for.
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight tensor.
        """
        if request.model_id not in llm_instances:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model_id} not found"
            )

        # Get the requested model
        llm = llm_instances[request.model_id]

        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(
            llm.collective_rpc,
            "update_named_param",
            args=(request.name, dtype, request.shape),
        )

        return {
            "message": f"Request received, updating named parameter for model {request.model_id}"
        }

    @app.post("/reset_prefix_cache/{model_id}")
    async def reset_prefix_cache(model_id: str):
        """
        Resets the prefix cache for the specified model.

        Args:
            model_id (`str`): The ID of the model to reset the prefix cache for.
        """
        if model_id not in llm_instances:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Get the requested model
        llm = llm_instances[model_id]

        success = llm.llm_engine.reset_prefix_cache()
        return {
            "message": f"Request received, resetting prefix cache for model {model_id}, status: {str(success)}"
        }

    @app.post("/close_communicator/{model_id}")
    async def close_communicator(model_id: str):
        """
        Closes the weight update group and cleans up associated resources for the specified model.

        Args:
            model_id (`str`): The ID of the model to close the communicator for.
        """
        if model_id not in llm_instances:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Get the requested model
        llm = llm_instances[model_id]

        llm.collective_rpc("close_communicator")
        return {
            "message": f"Request received, closing communicator for model {model_id}"
        }

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve",
            help="Run the vLLM serve script",
            dataclass_types=ScriptArguments,
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
