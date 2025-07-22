from variable_elim import *
from read_bayesnet import *

net = BayesNet('survey.bif')
print("Nodes:")
print(net.nodes)
print("Values:")
print(net.values)
print("Parents:")
print(net.parents)
print("Probabilities:")
print(net.probabilities)


ve = VariableElimination(net)

# Set the node to be queried as follows:
query = 'Transportation'

# The evidence is represented in the following way (can also be empty when there is no evidence):
observed = {'Burglary': 'True'}
heuristic = "min-weight"

# Determine your elimination ordering before you call the run function. The elimination ordering
# is either specified by a list or a heuristic function that determines the elimination ordering
# given the network. Experimentation with different heuristics will earn bonus points. The elimination
# ordering can for example be set as follows:

result_T = ve.run(query="T", observed={}, heuristic="min-weight")
print("P(T):")
print(result_T)


