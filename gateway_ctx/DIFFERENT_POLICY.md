This document is about how to enhance gateway_ctx to make it support different demotion/promotion policy

currently the policy is based on age, we want to support another policy which order agents based on current decode throughput of the agent. The decode throughput is the number of decoded token and sum of the time it spent in  llm request (including the queuing time)
- when we demotion, we pick the agent that has the highest throughput
- when we promtion, we pick the agent that has the lowest throughput


the per agent state may need to store the accumaled llm time and current queue start time, and the total completion time.
During demotion the throught doesn't consider the current queue time (because we only demote ongoing agent)
during prompotion however the current queuing time needs to be considered also

enhance the /ctx-aware/start with an optional mode selection argument. if the user does not provide we still use the previous age mode. the user can use 'throughput' to enter the above throughput mode.