### Call BuildNetwork()
### Input: Higher-order dependency rules
### Output: HON network


from collections import defaultdict, Counter

Graph = defaultdict(dict)
Verbose = False

def Initialize():
    Graph = defaultdict(dict)

def BuildNetwork(Rules):
    VPrint('Building network')
    # Initialize()
    Graph.clear()
    SortedSource = sorted(Rules, key=lambda x: len(x))
    for source in SortedSource:
        for target in Rules[source]:
            Graph[source][(target,)] = Rules[source][target]
            # following operations are destructive to Rules
            if len(source) > 1:
                Rewire(source, (target,))
    RewireTails()
    return Graph

def Rewire(source, target):
    PrevSource = source[:-1]
    PrevTarget = (source[-1],)
    if not PrevSource in Graph or not source in Graph[PrevSource]:
        Graph[PrevSource][source] = Graph[PrevSource][PrevTarget]
        del(Graph[PrevSource][PrevTarget])

def RewireTails():
    ToAdd = []
    ToRemove = []
    for source in Graph:
        for target in Graph[source]:
            if len(target) == 1:
                NewTarget = source + target
                while len(NewTarget) > 1:
                    if NewTarget in Graph:
                        ToAdd.append((source, NewTarget, Graph[source][target]))
                        ToRemove.append((source, target))
                        break
                    else:
                        NewTarget = NewTarget[1:]
    for (source, target, weight) in ToAdd:
        Graph[source][target] = weight
    for (source, target) in ToRemove:
        del(Graph[source][target])


###########################################
# Auxiliary functions
###########################################

def VPrint(string):
    if Verbose:
        print(string)
