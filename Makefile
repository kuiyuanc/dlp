.PHONY: ssh pull

from =
to =

ssh:
	ssh chenky2003@hpclogin02.cs.nycu.edu.tw

pull:
	scp -r chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/$(from) $(to)

push:
	scp -r $(from) chenky2003@hpclogin02.cs.nycu.edu.tw:/net/cs/110/110550035/$(to)
