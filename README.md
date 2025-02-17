# Backend

`torch.distributed` supports three built-in backends, each with different capabilities. The table below shows which functions are available for use with CPU / CUDA tensors. MPI supports CUDA only if the implementation used to build PyTorch supports it.

| Backend/Device | CPU (Gloo) | GPU (Gloo) | CPU (MPI) | GPU (MPI) | CPU (NCCL) | GPU (NCCL) |
|----------------|------------|------------|-----------|-----------|------------|------------|
| **send**       | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |
| **recv**       | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |
| **broadcast**  | ✓          | ✓          | ✓         | ?         | ✘          | ✓          |
| **all_reduce** | ✓          | ✓          | ✓         | ?         | ✘          | ✓          |
| **reduce**     | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |
| **all_gather** | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |
| **gather**     | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |
| **scatter**    | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |
| **reduce_scatter** | ✘      | ✘          | ✘         | ✘         | ✘          | ✓          |
| **all_to_all** | ✘          | ✘          | ✓         | ?         | ✘          | ✓          |
| **barrier**    | ✓          | ✘          | ✓         | ?         | ✘          | ✓          |



# 分布式通信原语简要说明及测试代码

1. **send**  
   - **功能**: 点对点通信，发送张量到目标进程。  
   - **用途**: 用于在分布式进程之间发送数据。  
   - **测试代码**: 
     ```python
     import torch
     import torch.distributed as dist

     dist.init_process_group("gloo", rank=0, world_size=2)
     tensor = torch.tensor([1, 2, 3])
     if dist.get_rank() == 0:
         dist.send(tensor=tensor, dst=1)
     elif dist.get_rank() == 1:
         recv_tensor = torch.empty(3)
         dist.recv(tensor=recv_tensor, src=0)
         print(recv_tensor)
     ```

2. **recv**  
   - **功能**: 点对点通信，接收来自其他进程的张量。  
   - **用途**: 用于从其他分布式进程接收数据。  
   - **测试代码**: 同 `send` 的测试代码，调用 `dist.recv` 接收数据。

3. **broadcast**  
   - **功能**: 从一个源进程将张量广播到所有其他进程。  
   - **用途**: 常用于将相同的数据分发给多个进程。  
   - **测试代码**: 
     ```python
     tensor = torch.tensor([0])
     if dist.get_rank() == 0:
         tensor += 1  # Only rank 0 modifies the tensor
     dist.broadcast(tensor, src=0)
     print(f"Rank {dist.get_rank()}: {tensor}")
     ```

4. **all_reduce**  
   - **功能**: 将所有进程中的张量按指定操作（如求和）进行归约，结果对所有进程可见。  
   - **用途**: 常用于全局统计计算（如总和、平均值）。  
   - **测试代码**: 
     ```python
     tensor = torch.tensor([dist.get_rank() + 1])
     dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
     print(f"Rank {dist.get_rank()}: {tensor}")
     ```

5. **reduce**  
   - **功能**: 将所有进程中的张量按指定操作归约，结果只保留在一个目标进程中。  
   - **用途**: 用于将计算结果收集到一个特定进程。  
   - **测试代码**: 
     ```python
     tensor = torch.tensor([dist.get_rank() + 1])
     dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
     if dist.get_rank() == 0:
         print(f"Reduced result: {tensor}")
     ```

6. **all_gather**  
   - **功能**: 将每个进程的张量收集并拼接到所有进程。  
   - **用途**: 用于共享每个进程的独立数据到所有其他进程。  
   - **测试代码**: 
     ```python
     tensor = torch.tensor([dist.get_rank()])
     gather_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
     dist.all_gather(gather_list, tensor)
     print(f"Rank {dist.get_rank()}: {gather_list}")
     ```

7. **gather**  
   - **功能**: 将所有进程的张量收集到一个目标进程。  
   - **用途**: 用于集中处理数据，例如在一个进程上保存所有结果。  
   - **测试代码**: 
     ```python
     tensor = torch.tensor([dist.get_rank()])
     if dist.get_rank() == 0:
         gather_list = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
     else:
         gather_list = None
     dist.gather(tensor, gather_list, dst=0)
     if dist.get_rank() == 0:
         print(f"Gathered result: {gather_list}")
     ```

8. **scatter**  
   - **功能**: 将一个进程的张量分发到多个进程，每个进程接收一部分数据。  
   - **用途**: 常用于将数据分片分发给不同的计算进程。  
   - **测试代码**: 
     ```python
     if dist.get_rank() == 0:
         scatter_list = [torch.tensor([i]) for i in range(dist.get_world_size())]
     else:
         scatter_list = None
     tensor = torch.empty(1)
     dist.scatter(tensor, scatter_list, src=0)
     print(f"Rank {dist.get_rank()}: {tensor}")
     ```

9. **reduce_scatter**  
   - **功能**: 先对所有进程的张量进行归约，再将归约结果分发到不同的进程。  
   - **用途**: 用于归约操作后立即分发结果，提高效率。  
   - **测试代码**: 
     ```python
     input = [torch.tensor([dist.get_rank() + 1]) for _ in range(dist.get_world_size())]
     output = torch.empty(1)
     dist.reduce_scatter(output, input, op=dist.ReduceOp.SUM)
     print(f"Rank {dist.get_rank()}: {output}")
     ```

10. **all_to_all**  
    - **功能**: 每个进程发送和接收等量的数据到所有其他进程。  
    - **用途**: 用于复杂的数据交换场景，例如分布式训练中的数据重新排列。  
    - **测试代码**: 
      ```python
      input = torch.tensor([dist.get_rank()] * dist.get_world_size())
      output = torch.empty(dist.get_world_size(), dtype=torch.int)
      dist.all_to_all(output, input)
      print(f"Rank {dist.get_rank()}: {output}")
      ```

11. **barrier**  
    - **功能**: 阻塞所有进程，直到所有进程都到达同步点。  
    - **用途**: 用于全局同步，确保所有进程完成当前任务后再继续。  
    - **测试代码**: 
      ```python
      print(f"Rank {dist.get_rank()} reached barrier.")
      dist.barrier()
      print(f"Rank {dist.get_rank()} passed barrier.")
      ```

