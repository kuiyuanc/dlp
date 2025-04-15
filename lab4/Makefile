train:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 train.sh

test:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 test.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	-rm -rf slurm-*.out

cleanall: clean
	-rm -rf saved_models
	-rm -rf logs
