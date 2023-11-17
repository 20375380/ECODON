import torch.cuda

class initailDevice:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.device_memory={}
        for i in range(self.num_gpus):
            self.device_memory[i]=torch.cuda.memory_allocated(i)
        self.sorted_gpus = sorted(self.device_memory.items(), key=lambda x: x[1])
        self.chosen_gpu = self.sorted_gpus[0][0]




