You are an expert Python programmer tasked with generating standardized function skeletons for competitive programming problems. Your goal is to analyze a given problem description and determine the most appropriate function signature (parameters and return type) to solve a *single instance* of the problem, ignoring any outer loops for multiple test cases.

You will be provided with:
1.  The full text description of the competitive programming problem, including Input and Output format specifications.
2.  The solution to the problem, which is usually a Python function/script.
3.  A list of predefined common structural patterns (clusters) with example Python function skeletons.

Your task is to:
1.  Carefully read the problem description, focusing on the `-----Input-----` and `-----Output-----` sections for a single instance/test case.
2.  Determine the essential inputs required to solve one instance (e.g., integers, strings, lists of integers, edge lists for graphs, grids).
3.  Determine the essential outputs produced for one instance (e.g., a single integer, a boolean, a list of integers, a tuple).
4.  Select the most appropriate cluster template from possible_clusters that best matches the problem's I/O structure.
5.  **Adapt** the chosen skeleton's function signature (`def solve(...) -> ...:`) to precisely match the identified inputs and outputs for the specific problem. Use accurate parameter names and Python type hints.
6.  Generate a basic docstring for the function explaining its purpose, arguments (`Args:`), and return value (`Returns:`).
7.  Ensure the function body contains only a placeholder like `pass` or `# TODO: Implement solution logic`. Do *not* include any input reading (`input()`) or output printing (`print()`) statements *inside* the function body. Do *not* include the outer test case loop (`for _ in range(T): ...`) *inside* the function.
8.  Provide a harness that takes a solution following the skeleton and a test case, and verifies whether the solution is correct. The harness should be able to handle the provided solution and the ground truth solution, in order to check for correctness of the candidate solution.

You should be following the following format:
<rationale>
... Your detailed reasoning for the function signature and docstring here ...
</rationale>
<skeleton>
```python
... The final, adapted Python function definition ...
```
</skeleton>

<harness_rationale>
... Your detailed reasoning for the harness here ...
</harness_rationale>
<harness>
```python
... The final, adapted Python harness ...
```
</harness>

Here's the problem description, solution, and possible clusters:

**Problem Description:**
```text
{problem}
```

**Solution:**
```python
{solution}
```

**Possible Cluster Skeletons:**
```text
{possible_clusters}
```
