from itertools import chain

## flatten list of lists
def flatten(inList):
    return(list(chain(*inList)))


