This discribe the addition SLO aware mode, which can be turned on in a similar way as the ctx aware mode, and similar to ctx aware, this is also based concurrency control (therefore it should be on a similar control path as the ctx aware)

SLO-aware mode requires ctx-aware mode to already be turned on. The user cannot turn on SLO-aware mode by itself.


To turn on the SLO-aware mode, the user needs to provide an SLO target in tokens/s, denoted as \(x\).

For each agent, we define its *SLO slack* using the throughput that is already stored in the agent metadata. In the current gateway, that stored throughput is `current_output_tokens_per_s`, which is computed from:

- `total_completion_tokens`
- `total_llm_request_duration_s`, which in `gateway_ctx` includes full
  end-to-end request time, including gateway-side wait while the agent is
  blocked in `pending` or `ralexation`

So in the formula below:

- \(y\) is the agent's stored `current_output_tokens_per_s`
- \(t\) is the corresponding `total_llm_request_duration_s`

Therefore, \(yt\) is exactly the completed decode work so far, i.e. `total_completion_tokens`. Under the target throughput \(x\), the time required to complete the same amount of work would be

$$
t' = \frac{yt}{x}.
$$

We define the SLO slack as the extra delay budget the agent still has relative to the target:

$$
\text{SLO slack} = t' - t = \frac{yt}{x} - t = t\left(\frac{y}{x} - 1\right).
$$

A larger positive slack indicates more room to trade performance for efficiency without violating the SLO, while zero or negative slack indicates that the agent is already at or below the target.

This definition intentionally reuses the throughput already tracked on each agent. It does not introduce a new instantaneous throughput metric for SLO-aware mode.


Besides the slo target, the user also need to provide a policy.
Now we provide two policies: "push-back-half-slack" and "push-back-80p-slack".

If the minimum stored throughput among all alive agents is smaller than the SLO, the policy will be triggered.

Whenever an ongoing agent returns from a request, if its stored throughput is larger than the average stored throughput of all active agents and the SLO target, we push it into a "ralexation" state. The ralexation time is half of its SLO slack for "push-back-half-slack", or 80% of its SLO slack for "push-back-80p-slack".

The "ralexation" state is similar to the normal pending state, but the wake-up condition is different. Normal pending is woken up by the regular scheduler admission logic, while "ralexation" is woken up by the SLO-aware policy.

If both pending and "ralexation" would apply to the same agent, normal pending takes priority.

When the "ralexation" wake-up condition is met, the agent does not always go directly back to ongoing. If there is already any normal pending agent, the "ralexation" agent should first transition into normal pending and then wait for the regular scheduler admission logic. Otherwise, it should be treated as if it were a new incoming agent: if the ctx-aware policy would keep it out of pending, make it ongoing; otherwise, make it pending.
