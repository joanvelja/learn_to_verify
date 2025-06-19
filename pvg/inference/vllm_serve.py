import argparse
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_available,
)

if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI


if is_pydantic_available():
    from pydantic import BaseModel, Field


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.config import PoolerConfig
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
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
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
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
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
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        task_type (`str`, *optional*, defaults to `"auto"`):
            The type of task to run on the model. If set to `"auto"`, the task type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: str | None = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: int | None = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )

    max_seq_len_to_capture: int | None = field(
        default=None,
        metadata={
            "help": "Maximum sequence length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode."
        },
    )
    task_type: str = field(
        default="auto",
        metadata={
            "help": "The type of task to run on the model. If set to `auto`, the task type will be automatically determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )


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
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_cls=WeightSyncWorker,
        task=script_args.task_type,
        override_pooler_config=(
            PoolerConfig(pooling_type="LAST", softmax=False, normalize=False)
            if script_args.task_type == "classify"
            else None
        ),
    )

    app = FastAPI()

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
        """
        Retrieves the tensor parallel size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the tensor parallel size.

        Example response:
        ```json
        {"tensor_parallel_size": 8}
        ```
        """
        return {"tensor_parallel_size": llm.llm_engine.parallel_config.tensor_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: str | None = None
        logprobs: int | None = None
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        stop: list[str] | None = None

    class ClassifyRequest(BaseModel):
        inputs: list[str] = Field(..., description="List of input strings to classify.")

    class ChatRequest(BaseModel):
        prompts: list[list[dict[str, str]]]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: str | None = None
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        stop: list[str] | None = None
        chat_template: str = ""
        continue_final_message: bool = True
        add_generation_prompt: bool = False
        use_tqdm: bool = True

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]
        logprobs: list[list[dict[int, float | None]]] | None = None

    class ClassifyResponse(BaseModel):
        scores: list[float] = Field(
            ...,
            description="List of classification scores corresponding to the inputs.",
        )

    class ChatResponse(BaseModel):
        completion_ids: list[list[int]]
        logprobs: list[list[dict[int, float | None]]] | None = None

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
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
            logprobs=request.logprobs,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            include_stop_str_in_output=True if request.stop else False,
            seed=42,
        )
        all_outputs = llm.generate(request.prompts, sampling_params=sampling_params)
        # completion_ids = [
        #     list(output.token_ids)
        #     for outputs in all_outputs
        #     for output in outputs.outputs
        # ]
        # return {"completion_ids": completion_ids}
        # --- Extract completion_ids AND logprobs ---
        completion_ids = []
        logprobs_data = [] if request.logprobs is not None else None

        # --- Add Detailed Logging ---
        raw_logprobs_for_logging = []

        for i, request_output in enumerate(all_outputs):
            for j, output in enumerate(request_output.outputs):
                completion_ids.append(list(output.token_ids))
                if request.logprobs is not None and output.logprobs is not None:
                    token_logprobs_list = []
                    raw_step_logprobs_list = []  # For logging

                    for k, step_logprobs in enumerate(output.logprobs):
                        raw_step_data = {}  # For logging
                        if step_logprobs:
                            current_step_dict: dict[int, float | None] = {}
                            for token_id, logprob_obj in step_logprobs.items():
                                # --- Log the raw value ---
                                raw_val = logprob_obj.logprob
                                raw_step_data[int(token_id)] = raw_val  # Store raw value for logging
                                # --------------------------

                                # Replace non-finite values with None for JSON compatibility
                                if not math.isfinite(raw_val):
                                    current_step_dict[int(token_id)] = None
                                else:
                                    # Check if it's exactly 0.0 - might indicate an issue or high prob
                                    if raw_val == 0.0:
                                        logger.debug(
                                            f"Logprob is exactly 0.0 for token {token_id} at step {k}, output {j}, request {i}"
                                        )
                                    current_step_dict[int(token_id)] = raw_val

                            token_logprobs_list.append(current_step_dict)
                            raw_step_logprobs_list.append(raw_step_data)  # Add raw data for this step
                        else:
                            token_logprobs_list.append({})
                            raw_step_logprobs_list.append({})  # Add empty raw data

                    logprobs_data.append(token_logprobs_list)
                    raw_logprobs_for_logging.append(raw_step_logprobs_list)  # Add raw data for this output

                elif request.logprobs is not None:
                    logprobs_data.append([])
                    raw_logprobs_for_logging.append([])  # Add empty raw data

        # --- Log the raw data before returning ---
        # Use json.dumps for potentially cleaner multi-line output if needed
        logger.info(
            f"Raw logprobs extracted (before None conversion): {json.dumps(raw_logprobs_for_logging, indent=2)}"
        )
        logger.info(f"Processed logprobs for response: {json.dumps(logprobs_data, indent=2)}")

        # --- Return both completion_ids and logprobs ---
        return (
            {"completion_ids": completion_ids, "logprobs": logprobs_data}
            if request.logprobs is not None
            else {"completion_ids": completion_ids}
        )

    @app.post("/chat/", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Send an instruction-tuned chat request to the model.

        Args:
            request (`ChatRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to chat.

        Returns:
            `ChatResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.
                - `logprobs` (list of list of dict of `int` to `float` or `None`): A list of lists of logprobs for each generated completion.
        """
        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
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
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            include_stop_str_in_output=True if request.stop else False,
            seed=42,
        )

        assert request.chat_template != "", "Mistaken chat template... Did not pass it correctly?"

        logger.info(f"[DEBUG]: request.prompts: {request.prompts}")

        all_outputs = llm.chat(
            messages=request.prompts,
            sampling_params=sampling_params,
            chat_template=request.chat_template,
            continue_final_message=request.continue_final_message,
            add_generation_prompt=request.add_generation_prompt,
            use_tqdm=request.use_tqdm,
        )

        # Extract completion_ids from vLLM RequestOutput objects
        completion_ids = []
        for request_output in all_outputs:
            for output in request_output.outputs:
                completion_ids.append(list(output.token_ids))

        return {"completion_ids": completion_ids}

    @app.post("/classify/", response_model=ClassifyResponse)
    async def classify(request: ClassifyRequest):
        """
        Classifies the provided inputs using the Reward Model.

        Args:
            request (`ClassifyRequest`):
                - `inputs` (list of `str`): A list of input strings for the model to classify.

        Returns:
            `ClassifyResponse`:
                - `scores` (list of `float`): A list of scores for each input string.

        Example request:
        ```json
        {"inputs": ["This is a good example.", "This is a bad one."]}
        ```

        Example response:
        ```json
        {"scores": [0.95, 0.12]}
        ```
        """
        if script_args.task_type != "classify":
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400,
                detail="Classification endpoint is only available when task_type is set to 'classify'.",
            )

        # Assuming llm.classify exists and follows the expected interface
        try:
            outputs = llm.classify(request.inputs)
            # Extract scores - adjust the exact path based on vLLM's classify output structure
            # This assumes the structure mentioned: outputs[i].outputs.probs[0]
            scores = [output.outputs.probs[0] for output in outputs]
            logger.info(f"Classification requested for {len(request.inputs)} inputs. Scores: {scores}")
            return {"scores": scores}
        except AttributeError:
            from fastapi import HTTPException

            logger.error(
                "The 'classify' method is not available on the LLM object. Ensure vLLM version supports it and the model is loaded correctly for classification."
            )
            raise HTTPException(
                status_code=500,
                detail="Classification method not found on the LLM object.",
            )
        except Exception as e:
            from fastapi import HTTPException

            logger.error(f"Error during classification: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during classification: {str(e)}",
            )

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest, background_tasks: BackgroundTasks):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(request.host, request.port, script_args.tensor_parallel_size + 1),
        )
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function is called this way: update_named_param(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(llm.collective_rpc, "update_named_param", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(
            llm.collective_rpc,
            "update_named_param",
            args=(request.name, dtype, request.shape),
        )

        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        success = llm.llm_engine.reset_prefix_cache()
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        llm.collective_rpc("close_communicator")
        return {"message": "Request received, closing communicator"}

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
