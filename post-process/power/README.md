The script here only applies to run result with power/power-log.jsonl.
if this does not exist, the script should not treat it as an error, but report normally like no power information is found

The calculation needs to truncate measurement happens before the job run starts.

This generate three results
1. the average/min/max power of the GPU
2. the total energy cost
3. a list of (timeoffset to the job run start, power) points (can be later used for visulation)