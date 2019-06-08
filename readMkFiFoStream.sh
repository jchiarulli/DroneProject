#!/bin/bash

if [ -p fifo264  ]
then
	rm fifo264
fi
mkfifo fifo264
nc -u -l -v -p 9000 > fifo264
