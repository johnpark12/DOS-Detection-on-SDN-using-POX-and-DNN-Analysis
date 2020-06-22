import math
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Meant purely to calculate entropy from a given window of packets.
# Meant to be general purpose, so will accept only a window of strings
def calculateEntropy(window):
    window = [x.toStr() for x in window]
    uniqueStrings = list(set(window))
    entropy = 0
    for s in uniqueStrings:
        p = float(window.count(s))/float(len(window))
        entropy += p * math.log(1/p,2)
    return entropy