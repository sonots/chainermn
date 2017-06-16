import chainer.cuda
import math
import mpi4py.MPI

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


class NonCudaAwareCommunicator(_base.NodeAwareCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NonCudaAwareCommunicator, self).__init__(mpi_comm, use_nccl=True)
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()
        self.cpu_buffer_a = _memory_utility.HostPinnedMemory()
        self.cpu_buffer_b = _memory_utility.HostPinnedMemory()

    def broadcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            data = param.data
            tmp_cpu = chainer.cuda.to_cpu(data)
            self.mpi_comm.Bcast(tmp_cpu)
            tmp_gpu = chainer.cuda.to_gpu(tmp_cpu)
            data[:] = tmp_gpu

    def allreduce_grad(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_elems_per_node = int(math.ceil(n_elems_total / self.inter_size))
        n_bytes_per_node = n_elems_per_node * itemsize
        n_bytes_buffer = n_bytes_per_node * self.inter_size

        self.gpu_buffer_a.assign(n_bytes_buffer)
        self.gpu_buffer_b.assign(n_bytes_buffer)
        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)

        # Intra-node reduce
        self.intra_nccl_comm.reduce(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(), n_elems_total,
            nccl.NCCL_FLOAT, nccl.NCCL_SUM, 0, stream.ptr)

        # Inter-node allreduce
        if self.intra_rank == 0:
            self.cpu_buffer_a.assign(n_bytes_buffer)
            self.cpu_buffer_b.assign(n_bytes_buffer)

            self.gpu_buffer_b.array(n_bytes_buffer).data.copy_to_host(
                self.cpu_buffer_b.ptr(), n_bytes_buffer)

            self.inter_mpi_comm.Allreduce(
                [self.cpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT],
                [self.cpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])

            arr = self.gpu_buffer_b.array(n_elems_total)
            arr.data.copy_from_host(self.cpu_buffer_a.ptr(), n_bytes_buffer)
            arr *= 1.0 / self.size

        # Intra-node bcast
        self.intra_nccl_comm.bcast(
            self.gpu_buffer_b.ptr(), n_elems_total, nccl.NCCL_FLOAT, 0,
            stream.ptr)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
