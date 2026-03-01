I would like to improve con-drive by making it compatibale with the config/port_profiles.toml strategy we use and the simplify the number of argument one need to supply in the job configuration toml

1. by default the con-driver use the gateway_parse_port to get llm service. so the user should only need to provide a port profile as an input

2. gateway = true, gateway_job_output_root = "gateway-output", gateway_timeout_s = 3600.0; make these by default 

3. the con-driver first query the model name from the endpoint. There should only be one model avaialbe, and this is used as the test model name


# for harbor backend (originally forwarded args)
some arguments we want to pass to harbor can be get from the top level based on some pattern
E.g. current in the example we have
```
# Forwarded directly to harbor trials start
forwarded_args = [
  "--agent", "terminus-2",
  "--model", "hosted_vllm/Qwen3-Coder-30B-A3B-Instruct",
  "--agent-kwarg", "api_base=http://127.0.0.1:11457/v1",
  "--agent-kwarg", "model_info={\"max_input_tokens\":32768,\"max_output_tokens\":8192,\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}",
  "--agent-kwarg", "trajectory_config={\"linear_history\":true}",
]
```

1. Please refer to agent_model_endpont_configuration.md to auto populate the llm endpoint information to the agent. use the coverall method to ensure that no matter what agent one is using, the api url is passed

2. the model name is always the one we get from the serving endpoint prefixed by hosted_vllm/

4. the configs/model_config.toml has context_window for each model. always pass --agent-kwarg, model_info={\"max_input_tokens\": context_window,\"max_output_tokens\": context_window,\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}

Take the current config.example.toml as and example


```
[driver]
# Required
driver_backend = "harbor"
pool = "swebench-verified"
max_concurrent = 500
n_task = 500
results_dir = "con-driver-test-3"
# add a port profile number here
# add a agent type here (in this example is terminus-2)

# Optional
pattern = "eager"     # eager | poisson | possion
harbor_bin = "harbor" # or "uv run harbor"
seed = 7
dry_run = false
sample_without_replacement = true


# vllm_log = true # this is defaulted, user may or may not provide
# vllm_log_endpoint = "http://127.0.0.1:12138/metrics" # the user required to provide the port profile number
# vllm_log_interval_s = 1.0 # this is defaulted, user may or may not provide
# vllm_log_timeout_s = 5.0 # this is defaulted, user may or may not provide

# Gateway mode (enabled by default)
# gateway = true # this is defaulted, user may or may not provide
# gateway_url = "http://127.0.0.1:11457" # the user required to provide the port profile number 
# note that for harbor backend use the gateway_parsed_port
# gateway_job_output_root = "gateway-output" # this is defaulted, user may or may not provide
# gateway_timeout_s = 3600.0 # this is defaulted, user may or may not provide


# all the information below can be inferred by the backend specific logic
# agent: a part of the required field
# model: we know from the inital probe, prepend the hosted_vllm/
# "--agent-kwarg", "api_base" this can be derived from the port profile
# model info is always passed. max_input_tokens and max_output_tokens are both set to context window (based on the model), token cost is always 0
#"--agent-kwarg", "trajectory_config={\"linear_history\":true}" this should be maintained in another toml file. where we specify for different agent want special config we want.
#
# Forwarded directly to harbor trials start
# forwarded_args = [
#  "--agent", "terminus-2",
#  "--model", "hosted_vllm/Qwen3-Coder-30B-A3B-Instruct",
#  "--agent-kwarg", "api_base=http://127.0.0.1:11457/v1",
#  "--agent-kwarg", "model_info={\"max_input_tokens\":32768,\"max_output_tokens\":8192,\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}",
#  "--agent-kwarg", "trajectory_config={\"linear_history\":true}",
#]

```

As you can see only the first part is necessary



# task
place accomplish we I described. 
port profile 1 is alive and you can use that to do some small scale validation
