train:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 train.sh

test:
	sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 test.sh

pull:
	scp -r chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/dll/lab4/logs E:\dll\lab4
	scp -r chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/dll/lab4/saved_models E:\dll\lab4
	scp -r chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/dll/lab4/data E:\dll\lab4

plot:
	pipenv run tensorboard --logdir logs

clean:
	-rm -rf slurm-*.out
	find . -type d -name "__pycache__" -exec rm -rf {} +

cleanall: clean
	-rm -rf saved_models logs data
