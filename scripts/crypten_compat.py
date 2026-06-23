"""CrypTen 0.4.1 与 PyTorch 2.8+ 的 InProcessCommunicator 兼容补丁。"""


def patch_crypten_inprocess():
    from crypten.communicator.in_process_communicator import InProcessCommunicator
    from torch.distributed import ReduceOp

    if getattr(InProcessCommunicator, "_phantomshield_patched", False):
        return

    _orig_ar = InProcessCommunicator.all_reduce
    _orig_rd = InProcessCommunicator.reduce

    def all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False, batched=False):
        if batched:
            return [_orig_ar(self, t, op=op, async_op=async_op) for t in tensor]
        return _orig_ar(self, tensor, op=op, async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, async_op=False, batched=False):
        if batched:
            return [_orig_rd(self, t, dst, op=op, async_op=async_op) for t in tensor]
        return _orig_rd(self, tensor, dst, op=op, async_op=async_op)

    InProcessCommunicator.all_reduce = all_reduce
    InProcessCommunicator.reduce = reduce
    InProcessCommunicator._phantomshield_patched = True
