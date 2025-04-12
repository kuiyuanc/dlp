train:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 --nodelist=cmpt100 train.sh

test:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 --nodelist=cmpt100 test.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	-rm -rf slurm-*.out

cleanall: clean
	-rm -rf checkpoints
	-rm -rf Lab4_template/logs
