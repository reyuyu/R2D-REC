import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from test_user_action_auxiliary import FakeTokenizer, make_args, make_sample
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from llamafactory.data.action_select import ActionSelectMetadataParser
from llamafactory.train.sft.user_action_auxiliary import ACTION_STAT_SIZE, UserActionAuxiliaryController


class TinyDistributedCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 16)
        self.head = nn.Linear(16, 64, bias=False)
        self.forward_count = 0

    def forward(self, input_ids, labels):
        self.forward_count += 1
        logits = self.head(self.embedding(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(logits=logits, loss=loss)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(23)
    device = torch.device("cuda", local_rank)
    model = DistributedDataParallel(TinyDistributedCausalLM().to(device), device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sample = make_sample(answer=((10, 20, 30, 40),) if local_rank else ((10, 20, 30, 40), (10, 20, 30, 41)))
    metadata = ActionSelectMetadataParser(FakeTokenizer()).parse(sample["input_ids"], sample["labels"])
    input_ids = torch.tensor([sample["input_ids"]], device=device)
    labels = torch.tensor([sample["labels"]], device=device)
    controller = UserActionAuxiliaryController(
        FakeTokenizer(), make_args(user_action_aux_vectorized_enabled=True)
    )
    outputs = model(input_ids=input_ids, labels=labels)
    result = controller.compute(outputs.logits, labels, [metadata], outputs.loss, 100)
    ((outputs.loss + result.loss) / dist.get_world_size()).backward()
    statistics = torch.zeros(5 + ACTION_STAT_SIZE, device=device)
    statistics[0], statistics[1], statistics[5:] = (outputs.loss + result.loss).detach(), 1, result.statistics
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    optimizer.step()
    assert model.module.forward_count == 1
    assert torch.isfinite(statistics).all()
    flat = torch.cat([parameter.detach().flatten() for parameter in model.module.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    assert all(torch.allclose(gathered[0], item, atol=1e-6, rtol=1e-6) for item in gathered[1:])
    dist.barrier()
    if local_rank == 0:
        print("PASS: two-rank DDP vectorized Action auxiliary forward/backward/all-reduce/optimizer smoke")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
