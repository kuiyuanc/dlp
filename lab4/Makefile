train:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 train.sh

test:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 test.sh

clean:
	-rm -rf slurm-*.out
	find . -type d -name "__pycache__" -exec rm -rf {} +

cleanall: clean
	-rm -rf saved_models logs data
