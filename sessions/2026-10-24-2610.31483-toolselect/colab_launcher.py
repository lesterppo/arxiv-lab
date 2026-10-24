import subprocess, sys, os
p = subprocess.Popen(
    [sys.executable, '/content/colab_rl_toolselect.py'],
    stdout=open('/content/ts_run.log', 'w'),
    stderr=subprocess.STDOUT,
    start_new_session=True, cwd='/content',
    env={**os.environ, 'TS_MODEL': 'Qwen/Qwen3-8B', 'TS_RL_STEPS': '60', 'TS_N_ROLLOUT': '4'})
print(f'PID:{p.pid}|SESSION:ts1024|LOG:/content/ts_run.log')
