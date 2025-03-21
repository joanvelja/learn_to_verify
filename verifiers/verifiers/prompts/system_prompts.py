SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

VERIFY_PROMPT = """
You are a skilled answer checker. You will be given a problem within <problem> tags and a candidate solution within <solution> tags. Refer to the problem. Your role is to verify whether the solution contains mistakes.
Start by breaking down the provided solution into steps. Within <verification> tags, check if the maths in each step is correct. Assess the correctness of the solution by checking the solution provided on a step-by-step basis. You should not solve the problem yourself, but only verify the solution provided by the prover. Only after you have verified the solution step by step, state whether the solution is correct or incorrect within <answer> tags. If you encounter an error at any point, end the verification process and state that the solution is incorrect.

The format you will receive the problem and solution in is as follows:

<problem>
...
</problem>

<solution>
...
</solution>

The format you need to return the verification in is as follows:

<verification>
...
</verification>

<answer>
Correct OR Incorrect
</answer>
"""