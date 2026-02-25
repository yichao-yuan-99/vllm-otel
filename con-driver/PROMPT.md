implement a concurrent driver to launch harbor jobs in parallel

the input:
- a user provide harbor trails start with all the options except the `-p` which points to the dataset. This is in a arg `--harbor-args` and provided in a form like `--harbor-args="--optiona=a,--optionb=b"`
- instead the user provides a `--pool` arguement which include a list of harbor dataset, like `--pool="swebench-verified,terminal-bench@2.0"`
- arrival pattern `--pattern=eager/possion`
- pattern parameter like `--pattern-args="--arga=a"` (e.g. to control poisson process)
- max concurrent `--max-concurrent` which limit the maximum concurrent jobs we want to run
- `--n-task` how many task we want to launch
- `--results-dir` where all the results should be contained

The driver launch tasks in the task pool specified by the datasets.
If it needs to launch a task, it uniformly sample one from the the pool.
The launching timing follows the arrival pattern.
- the driver first download the datasets with harbor and create a pool of all tasks in all the datasets
- we give each trial a unique id and a unique `--trials-dir` based on that id to avoid name conflicts
- the driver sample and lanuch jobs


We want a modular design: i.e. the lanuching mechanism should be seperate from the harbor command formation and semantics.
the harbor is like one of the backend we want to support.

A reference command for `harbor trails run` is like
```
harbor trials start   -p datasets/TesWN8yxgXDngAVxcKSxhJ/pytest-dev__pytest-10356/   --agent terminus-2   --model hosted_vllm/Qwen3-Coder-30B-A3B-Instruct   --agent-kwarg api_base=http://127.0.0.1:11111/v1   --agent-kwarg 'model_info={"max_input_tokens":32768,"max_output_tokens":8192,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'   --agent-kwarg 'trajectory_config={"linear_history":true}'   --trials-dir temp-t-2/temp
```


You can use localhost:12138 to do small scale test (use no more than 10 task to test)


create a concise design and write a README.md.
write a METHOD.md to explain the design itself.

you can use the .venv as the virtual env