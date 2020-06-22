import sys
from os import popen
from scapy.all import send, sendp, IP, UDP, Ether, TCP
import random
from time import sleep
import logging
logging.basicConfig(level=logging.INFO)

# Meant to be run on all hosts seperately to simulate "normal" UDP based traffic
# Provided the host IP and the number of hosts on the network as two args.
# Subnet size and number of other hosts must be pre-configured. Assumption is that hosts will be created in sequential order
# Provided the total number of packets to send as a second arg (Actually, using other defaults for now).
# These packets are send as batches to a random address from the destination list.
# There will be an X second pause between each batch.
sourceIP = sys.argv[1]
subnet = "10.0.0.1/24"
numberOfHosts = int(sys.argv[2])

totalPacketCount = 100000
sleepTime = 1
batchSize = 10

# Generate a list of destination ips 
subAddr, subSize = subnet.split("/")
if int(subSize)%8 != 0: raise Exception("Please make subnet addr a multiple of 8")
addrTemplate = ".".join(subAddr.split(".")[:int(subSize)//8])
print("Using addrTemplate {}".format(addrTemplate))
# print([x for x in range(1,numberOfHosts+1)])
allDestAddrs = [addrTemplate + "." + str(x) for x in range(1,numberOfHosts+1)]
destAddrs =  [x for x in allDestAddrs if x != sourceIP]
logging.info("Sending to addresses {}".format(destAddrs))

# Allocation of packets into batches of 
# for i in range(totalPacketCount//batchSize):
while True:
    destAddr = random.choice(destAddrs)
    # packets = Ether()/IP(dst=destAddr,src=sourceIP)/UDP(dport=80,sport=2)
    # packets = [IP(dst=destAddr,src=sourceIP)/UDP(dport=80,sport=2) for i in range(batchSize)]
    packets = [IP(dst=random.choice(destAddrs),src=sourceIP) / UDP(dport=80,sport=2) / "Benign" for i in range(batchSize)]
    logging.info(repr(packets))
    # logging.info(i)
    send(packets)
    # sendp(packets,iface="eth0",inter=0.1)
    sleep(sleepTime)

# TODO
# These packets are send on a random basis that alternates between burst and continuous towards random targets
# In general, "connections" are maintained until the number of packets 
# Multithread the packet sending both for performance reasons and in order to generate more natural traffic.