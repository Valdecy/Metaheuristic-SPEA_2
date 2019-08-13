# Metaheuristic-SPEA_2
SPEA 2 (Strength Pareto Evolutionary Algorithm 2) - Function to Minimize Multiple Objectives with Continuous Variables. Real Values Encoded. The function returns: 1) An array containing the used value(s) for each function and the output for each function f(x). For example, if the functions f(x1, x2) and g(x1, x2) are used in this same order, then the array would be [x1, x2, f(x1, x2), g(x1, x2)].  


* population_size = The population size. The Default Value is 5.

* archive_size = The archive size. It is an external set of individuals that is formed from the non-dominated solutions. The Default Value is 5.

* mutation_rate = Chance to occur a mutation operation. The Default Value is 0.1

* eta = Value of the mutation operator. The Default Value is 1.

* min_values = The minimum value that the variable(s) from a list can have. The default value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The default value is  5.

* generations = The total number of iterations. The Default Value is 50.

* list_of_functions = A list of functions. The default value is two fucntions [func_1, func_2].

* mu = Value of the breed operator. The Default Value is 1.

Kursawe Function Example:

<p align="center"> 
<img src="https://github.com/Valdecy/Metaheuristic-SPEA_2/blob/master/Python-MH-SPEA-2.gif">
</p>

# Acknowledgement 
This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Wei Chen (07.AUGUST.2019) - AFRL Summer Intern/Rising Senior at Stony Brook University.
