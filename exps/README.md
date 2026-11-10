1. Define an experiment YAML in this directory.
2. Generate its SLURM script with `python ../generate_sbatch_from_yaml.py <yaml> [output.sh]`.
3. Submit with `sbatch <output.sh>`; the H200/`--aicomputing` flow is deferred—see `world_model/docs/h200_kt_cloud_job_submission_runbook.md`.
