# Cluster execution

The template in this directory is scheduler- and institution-neutral. Configure
the repository, Conda environment, partition, account, and GPU constraint in
your own submission wrapper; do not commit site-specific usernames or paths.

Example:

```bash
sbatch \
  --export=ALL,DEXLINK_ROOT="$PWD",DEXLINK_CONDA_ENV=genesis,DEXLINK_COMMAND='dexlink-train --preset seq_oakink_v3_02_full_method' \
  cluster/slurm_job.sh
```

The script deliberately sources `.bashrc` before enabling Bash nounset because
some system shell initializers read variables before defining them.
