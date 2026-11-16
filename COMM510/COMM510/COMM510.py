from pymoo.algorithms.moo.nsga3 import NSGA3


from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

D = [5, 10, 100]
M = [3, 5, 10]

# create the reference directions to be used for the optimization
ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=12)

# create the algorithm object
algorithm = NSGA3(pop_size=2000,
                  ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("dtlz3", n_var=100, n_obj=5),
               algorithm,
               seed=1,
               termination=('n_gen', 60))

Scatter().add(res.F).show()
