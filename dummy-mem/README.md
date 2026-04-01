# Dummy GPU Memory Holder

`occupy_gpu_memory.py` allocates dummy GPU memory with PyTorch, then sleeps
until you interrupt it. On `Ctrl+C` or `SIGTERM`, it releases the allocation
and exits.

The main input is `--size`, in GiB. For example, `--size 10` allocates about
10 GiB of dummy memory on the selected CUDA device.

## Usage

```bash
python3 dummy-mem/occupy_gpu_memory.py --size 10
```

Choose a different GPU with `--device`:

```bash
python3 dummy-mem/occupy_gpu_memory.py --size 10 --device cuda:1
```

## Notes

- The script allocates a `torch.uint8` tensor so the requested size maps closely
  to raw bytes on the GPU.
- PyTorch with CUDA must be installed in the environment.
- If the requested size is larger than currently free GPU memory, the script
  exits with an error instead of partially allocating.
