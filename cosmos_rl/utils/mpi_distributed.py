import torch.distributed as dist
from mpi4py import MPI
import os
import torch
import random
from tensorrt_llm._utils import mpi_broadcast

from cosmos_rl.utils.constant import COSMOS_HTTP_RETRY_CONFIG
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.network_util import find_available_port

OMPI_COMM_TYPE_HOST = 9

global_comm = MPI.COMM_WORLD


def set_mpi_comm(new_comm):
    """
    Set the MPI communicator to be used by the distributed package.
    """
    global global_comm
    global_comm = new_comm


def mpi_comm():
    return global_comm


local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)


local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)


def mpi_rank():
    return mpi_comm().Get_rank()


def global_mpi_rank():
    return MPI.COMM_WORLD.Get_rank()


def global_mpi_size():
    return MPI.COMM_WORLD.Get_size()


def mpi_world_size():
    return mpi_comm().Get_size()


def local_mpi_rank():
    return local_comm.Get_rank()


def local_mpi_size():
    return local_comm.Get_size()


def init_distributed_with_MPI():
    if dist.is_initialized():
        return

    # FIXME: (lms) Support multi-nodes.
    local_rank = mpi_rank()
    global_rank = global_mpi_rank()
    world_size = global_mpi_size()

    cosmos_world_size = os.environ.get("COSMOS_WORLD_SIZE", None)
    cosmos_local_world_size = os.environ.get("COSMOS_LOCAL_WORLD_SIZE", None)

    assert cosmos_world_size is not None, "COSMOS_WORLD_SIZE is not set."
    assert cosmos_local_world_size is not None, "COSMOS_LOCAL_WORLD_SIZE is not set."

    if cosmos_world_size is not None:
        assert world_size == int(
            cosmos_world_size
        ), "COSMOS_WORLD_SIZE is not consistent with the world size of MPI."
    if cosmos_local_world_size is not None:
        assert mpi_world_size() == int(
            cosmos_local_world_size
        ), "COSMOS_LOCAL_WORLD_SIZE is not consistent with the local world size of MPI."

    rdzv_endpoint = os.environ.get("COSMOS_RDZV_ENDPOINT", None)
    assert rdzv_endpoint is not None, "COSMOS_RDZV_ENDPOINT is not set."
    rdzv_host, rdzv_port = rdzv_endpoint.split(":")

    torch.cuda.set_device(local_rank)
    max_port = 65535
    min_port = 12371

    for _ in range(COSMOS_HTTP_RETRY_CONFIG.max_retries):
        if not int(rdzv_port):
            rdzv_port = None
            if mpi_rank() == 0:
                rdzv_port = find_available_port(
                    start_port=random.randint(min_port, max_port)
                )
            rdzv_port = mpi_broadcast(rdzv_port, root=0)

        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = rdzv_host
        os.environ["MASTER_PORT"] = str(rdzv_port)

        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

        try:
            # We use nccl and gloo backend
            dist.init_process_group(
                "cuda:nccl,cpu:gloo",
                world_size=world_size,
                rank=local_rank,
                init_method=init_method,
            )
        except dist.DistNetworkError:
            continue
        if dist.is_initialized():
            break
    else:
        raise RuntimeError(
            f"Failed to initialize distributed environment after {COSMOS_HTTP_RETRY_CONFIG.max_retries} retries."
        )

    logger.info(
        f"[Rollout] init torch distributed environment inside trtllm worker with tcp://{rdzv_host}:{rdzv_port} in rank {local_rank}."
    )
