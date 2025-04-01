train:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 --nodelist=cmpt100 train.sh

test:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 --nodelist=cmpt100 test.sh

pull:
	scp -r chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/dlp/lab3/logs e:/dlp/lab3
	scp -r chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/dlp/lab3/saved_models e:/dlp/lab3

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	-rm -rf slurm-*.out

cleanval:
	-rm -rf test_results mask_scheduling imga

cleanall: clean cleanval
	-rm -rf saved_models logs
