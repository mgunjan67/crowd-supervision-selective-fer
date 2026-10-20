"""The prefetch adapter must retain every epoch's reference sample sequence."""
import torch
from torch.utils.data import DataLoader
from test_memory_safe import Indices
from run_prefetched import PrefetchedLoader

def test_prefetched_order_matches_original():
    dataset=Indices()
    a,b=torch.Generator(),torch.Generator()
    original=DataLoader(dataset,batch_size=8,shuffle=True,generator=a,num_workers=2,persistent_workers=True)
    adapter=PrefetchedLoader(dataset,batch_size=8,shuffle=True,generator=b)
    for epoch in (1,2,3):
        dataset.epoch.fill_(epoch)
        a.manual_seed(17+epoch*100003);b.manual_seed(17+epoch*100003)
        assert torch.equal(torch.cat(list(original)),torch.cat(list(adapter)))
    original._iterator._shutdown_workers()
