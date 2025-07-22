import torch.distributed as dist
from mpi4py import MPI
import os
import torch

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


def init_distributed_with_MPI(rdzv_host: str, rdzv_port: str):
    # FIXME: (lms) Support multi-nodes.
    local_rank = mpi_rank()
    world_size = mpi_world_size()

    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = rdzv_host
    os.environ["MASTER_PORT"] = rdzv_port

    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

    torch.cuda.set_device(local_rank)

    # We use nccl and gloo backend
    dist.init_process_group(
        "cuda:nccl,cpu:gloo",
        world_size=world_size,
        rank=local_rank,
        init_method=init_method,
    )
