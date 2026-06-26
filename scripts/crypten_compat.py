"""CrypTen 0.4.1 与 PyTorch 2.8+ 的 InProcessCommunicator 兼容补丁。"""


def patch_crypten_inprocess():
    from crypten.communicator.in_process_communicator import InProcessCommunicator
    from torch.distributed import ReduceOp
    import torch

    if getattr(InProcessCommunicator, "_phantomshield_patched", False):
        return

    _orig_ar = InProcessCommunicator.all_reduce
    _orig_rd = InProcessCommunicator.reduce
    _orig_reduce_op = InProcessCommunicator._reduce_op_to_function

    def _reduce_op_to_function(self, op):
        if op == ReduceOp.BXOR:
            def bxor(values, dim=0):
                vals = values.unbind(dim)
                out = vals[0].clone()
                for value in vals[1:]:
                    out = torch.bitwise_xor(out, value)
                return out
            return bxor
        if op == ReduceOp.BAND:
            def band(values, dim=0):
                vals = values.unbind(dim)
                out = vals[0].clone()
                for value in vals[1:]:
                    out = torch.bitwise_and(out, value)
                return out
            return band
        if op == ReduceOp.BOR:
            def bor(values, dim=0):
                vals = values.unbind(dim)
                out = vals[0].clone()
                for value in vals[1:]:
                    out = torch.bitwise_or(out, value)
                return out
            return bor
        return _orig_reduce_op(self, op)

    def all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False, batched=False):
        if batched:
            return [_orig_ar(self, t, op=op, async_op=async_op) for t in tensor]
        return _orig_ar(self, tensor, op=op, async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, async_op=False, batched=False):
        if batched:
            return [_orig_rd(self, t, dst, op=op, async_op=async_op) for t in tensor]
        return _orig_rd(self, tensor, dst, op=op, async_op=async_op)

    InProcessCommunicator._reduce_op_to_function = _reduce_op_to_function
    InProcessCommunicator.all_reduce = all_reduce
    InProcessCommunicator.reduce = reduce
    InProcessCommunicator._phantomshield_patched = True


def patch_crypten_windows_multiprocess():
    """Allow CrypTen DistributedCommunicator on Windows when Gloo is available."""

    import random
    import string

    import torch.distributed as dist
    from crypten.communicator.distributed_communicator import DistributedCommunicator

    if getattr(DistributedCommunicator, "_phantomshield_windows_mp_patched", False):
        return
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed Gloo backend is not available")

    def initialize(cls, rank, world_size, init_ttp=False):
        import os

        randomized_path = "crypten-".join(
            random.choice(string.ascii_letters) for _ in range(10)
        )
        default_args = {
            "DISTRIBUTED_BACKEND": "gloo",
            "RENDEZVOUS": f"file:///tmp/{randomized_path}",
            "WORLD_SIZE": world_size,
            "RANK": rank,
        }
        for key, val in default_args.items():
            if key not in os.environ:
                os.environ[key] = str(val)

        cls.instance = DistributedCommunicator(init_ttp=init_ttp)

    DistributedCommunicator.initialize = classmethod(initialize)
    DistributedCommunicator._phantomshield_windows_mp_patched = True
