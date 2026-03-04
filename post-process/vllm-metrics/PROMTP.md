I would like to implement a script to extract the guage and counter's diff for a vllm-log/ record of a run

Inputs:
- the run result root directory.
We can find the .tar.gz files in its vllm-log/ subdirectory

What to do:
In the input, we have the records at each time stamp.
In the output we want for each metrics we all its value in a structure.
It contains three equal sized lists.
one list is the original captured_at time, 
the other list is the value 
the last one is also the time but normalized to the first vllm-log record's time

the metrics we care are the ones with guage and counter type and with the name begins with vllm.

for counter type we, want to output the difference of v_t - v_{t-1} as the value.
I.e. if the first counter value is 5, second counter value is 10, then we output 0 (always 0 as here is no value before it) as the first value and ouptut 5 (10 - 5) as the second value.

each metric now includes the name, the type and the three list

Output directory is in the original job result directory's new subdirectory post-processed/vllm-log/


You can use  tests/output/con-driver/job-20260301T014756Z/ to test