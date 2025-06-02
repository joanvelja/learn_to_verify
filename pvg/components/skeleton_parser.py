# skeleton_parser.py

import ast
import inspect
import re
from typing import Callable, Optional, Tuple


def _params_match(sig, expected):
    # ignore defaults & annotations, allow extra *OPTIONAL* params
    required = [
        p.name
        for p in sig.parameters.values()
        if p.default is p.empty and p.kind == p.POSITIONAL_OR_KEYWORD
    ]
    return required[: len(expected)] == expected  # order preserved


class SkeletonParser:
    """Robust parser for extracting and accessing functions from skeletons"""

    def __init__(self, globals_dict: dict):
        self.globals_dict = globals_dict

    def extract_function_info(
        self, skeleton: str
    ) -> Optional[Tuple[Optional[str], str, list]]:
        """
        Extract function info from skeleton (standalone or class method).
        Returns (class_name_or_None, function_name, param_names) or None if not found.
        """
        # Method 1: Try AST parsing
        try:
            result = self._extract_with_ast(skeleton)
            if result:
                return result
        except Exception:
            pass

        # Method 2: Regex fallback
        try:
            result = self._extract_with_regex(skeleton)
            if result:
                return result
        except Exception:
            pass

        return None

    def extract_class_method_info(
        self, skeleton: str
    ) -> Optional[Tuple[str, str, list]]:
        """
        Backward compatibility method - extract class name, method name, and parameters.
        Returns (class_name, method_name, param_names) or None if not a class method.
        """
        result = self.extract_function_info(skeleton)
        if result and result[0] is not None:  # Has class name
            return result
        return None

    def _extract_with_ast(
        self, skeleton: str
    ) -> Optional[Tuple[Optional[str], str, list]]:
        """Extract using AST parsing - handles both standalone functions and class methods"""
        # Fix incomplete skeleton for AST parsing
        fixed_skeleton = self._fix_skeleton_for_ast(skeleton)

        try:
            tree = ast.parse(fixed_skeleton)
        except SyntaxError:
            return None

        class_name = None
        function_info = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                param_names = [arg.arg for arg in node.args.args]

                # For class methods, remove 'self' parameter
                if class_name and param_names and param_names[0] == "self":
                    param_names = param_names[1:]

                function_info = (class_name, function_name, param_names)
                # For standalone functions, return immediately
                # For class methods, continue to potentially find more methods
                if class_name is None:
                    break

        return function_info

    def _extract_with_regex(
        self, skeleton: str
    ) -> Optional[Tuple[Optional[str], str, list]]:
        """Extract using regex patterns - handles both standalone functions and class methods"""
        lines = [line.strip() for line in skeleton.split("\n") if line.strip()]

        class_name = None
        function_info = None

        class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        function_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^)]*)\s*\)"

        for line in lines:
            # Look for class definition
            class_match = re.search(class_pattern, line)
            if class_match:
                class_name = class_match.group(1)
                continue

            # reset when we leave a class block (blank line or dedent)
            if not line.startswith(("def", "@")) and line[0].isalpha():
                class_name = None

            # Look for function definition
            function_match = re.search(function_pattern, line)
            if function_match:
                function_name = function_match.group(1)
                params_str = function_match.group(2).strip()

                # Parse parameters
                if not params_str:
                    param_names = []
                else:
                    param_names = []
                    for param in params_str.split(","):
                        param = param.strip()
                        if param:
                            # Remove type hints and default values
                            param = param.split(":")[0].split("=")[0].strip()
                            param_names.append(param)

                # For class methods, remove 'self' parameter
                if class_name and param_names and param_names[0] == "self":
                    param_names = param_names[1:]

                function_info = (class_name, function_name, param_names)

                # For standalone functions, we're done
                if class_name is None:
                    break

        return function_info

    def _fix_skeleton_for_ast(self, skeleton: str) -> str:
        """Fix incomplete skeleton to make it parseable by AST"""
        lines = skeleton.split("\n")
        fixed_lines = []

        for line in lines:
            fixed_lines.append(line)

            # Add colon if missing
            stripped = line.strip()
            if (
                stripped.startswith("class ") or stripped.startswith("def ")
            ) and not stripped.endswith(":"):
                fixed_lines[-1] += ":"

            # Add pass statement after class/def lines
            if stripped.endswith(":") and (
                stripped.startswith("class ") or stripped.startswith("def ")
            ):
                indent = len(line) - len(line.lstrip()) + 4
                fixed_lines.append(" " * indent + "pass")

        return "\n".join(fixed_lines)

    def get_function_callable(self, skeleton: str) -> Optional[Callable]:
        """
        Get the actual callable function/method from globals based on skeleton.
        Returns a callable that can be used directly.
        Works for both standalone functions and class methods.
        """
        info = self.extract_function_info(skeleton)
        if not info:
            return None

        class_name, function_name, expected_params = info

        if class_name is None:
            # Standalone function
            return self._find_standalone_function(function_name, expected_params)
        else:
            # Class method
            matching_class = self._find_matching_class(
                class_name, function_name, expected_params
            )
            if not matching_class:
                return None

            # Try multiple instantiation strategies
            try:
                # Strategy 1: No-argument constructor
                instance = matching_class()
                return getattr(instance, function_name)
            except Exception:
                # Strategy 2: Try with None arguments (common pattern)
                try:
                    sig = inspect.signature(matching_class.__init__)
                    params = list(sig.parameters.keys())[1:]  # Remove 'self'
                    args = [None] * len(params)  # Try with None values
                    instance = matching_class(*args)
                    return getattr(instance, function_name)
                except Exception:
                    # Strategy 3: Static method check
                    try:
                        method = getattr(matching_class, function_name)
                        if isinstance(method, staticmethod):
                            return method.__func__
                        # Strategy 4: Return unbound method with wrapper
                        return lambda *args, **kwargs: method(
                            matching_class(), *args, **kwargs
                        )
                    except Exception:
                        return None

    def _find_standalone_function(self, function_name, expected_params):
        return self.globals_dict.get(function_name, None)

    def get_method_callable(self, skeleton: str) -> Optional[Callable]:
        """Backward compatibility - alias for get_function_callable"""
        return self.get_function_callable(skeleton)

    def get_function_name_info(
        self, skeleton: str
    ) -> Optional[Tuple[Optional[str], str]]:
        """
        Get the actual function info found in globals.
        Returns (class_name_or_None, function_name) for use with:
        - Standalone: g['function_name']
        - Class method: g['ClassName']().method_name
        """
        info = self.extract_function_info(skeleton)
        if not info:
            return None

        class_name, expected_function_name, expected_params = info

        if class_name is None:
            # Standalone function - find in globals
            for name, obj in self.globals_dict.items():
                if (
                    not name.startswith("__")
                    and callable(obj)
                    and not inspect.isclass(obj)
                ):
                    if (
                        name == expected_function_name
                        and self._function_matches_signature(obj, expected_params)
                    ):  # Iff the name matches, and the signature matches, then return the name
                        return None, name
        else:
            # Class method - find the actual class name in globals
            for name, obj in self.globals_dict.items():
                if not name.startswith("__") and inspect.isclass(obj):
                    if hasattr(obj, expected_function_name):
                        method = getattr(obj, expected_function_name)
                        if self._method_matches_signature(method, expected_params):
                            return name, expected_function_name

        return None

    def get_class_and_method_names(self, skeleton: str) -> Optional[Tuple[str, str]]:
        """
        Backward compatibility - get class and method names (class methods only).
        Returns (actual_class_name, method_name) for use with g['ClassName']().method_name
        """
        result = self.get_function_name_info(skeleton)
        if result and result[0] is not None:  # Has class name
            return result
        return None

    def _find_matching_class(
        self, expected_class_name: str, expected_method_name: str, expected_params: list
    ) -> Optional[type]:
        """Find class in globals that matches the expected signature"""
        # First try exact class name match
        if expected_class_name in self.globals_dict:
            obj = self.globals_dict[expected_class_name]
            if inspect.isclass(obj) and hasattr(obj, expected_method_name):
                method = getattr(obj, expected_method_name)
                if self._method_matches_signature(method, expected_params):
                    return obj

        # Then try any class with matching method signature
        for name, obj in self.globals_dict.items():
            if not name.startswith("__") and inspect.isclass(obj):
                if hasattr(obj, expected_method_name):
                    method = getattr(obj, expected_method_name)
                    if self._method_matches_signature(method, expected_params):
                        return obj

        return None

    def _method_matches_signature(
        self, method: Callable, expected_params: list
    ) -> bool:
        """Check if method signature matches expected parameters"""
        try:
            sig = inspect.signature(method)
            # method_params = list(sig.parameters.keys())

            # # Remove 'self' parameter for comparison
            # if method_params and method_params[0] == "self":
            #     method_params = method_params[1:]

            # return method_params == expected_params
            return _params_match(sig, expected_params)
        except Exception:
            return False

    def _function_matches_signature(
        self, func: Callable, expected_params: list
    ) -> bool:
        """Check if standalone function signature matches expected parameters"""
        try:
            sig = inspect.signature(func)
            # func_params = list(sig.parameters.keys())
            # return func_params == expected_params
            return _params_match(sig, expected_params)
        except Exception:
            return False

    def create_verification_call(
        self, skeleton: str, verify_function_name: str = "verify_solution"
    ) -> Optional[str]:
        """
        Create the verification call string for use with eval/exec.
        Returns string like:
        - For standalone functions: g['verify_solution'](g['function_name'])
        - For class methods: g['verify_solution'](g['Solution']().methodName)
        """
        result = self.get_function_name_info(skeleton)
        if not result:
            return None

        class_name, function_name = result

        if class_name is None:
            # Standalone function
            return f"g['{verify_function_name}'](g['{function_name}'])"
        else:
            # Class method
            return f"g['{verify_function_name}'](g['{class_name}']().{function_name})"

    def verify_external_candidate(
        self,
        skeleton: str,
        candidate_code: str,
        verify_function_name: str = "verify_solution",
    ) -> Optional[tuple]:
        """
        Verify an external candidate solution against the skeleton.

        Args:
            skeleton: The function skeleton/signature to match
            candidate_code: String containing the candidate solution code
            verify_function_name: Name of verification function in globals_dict

        Returns:
            Result of verify_function(candidate_function) or None if extraction fails
        """
        # Get verification function
        verify_func = self.globals_dict.get(verify_function_name)
        if not verify_func or not callable(verify_func):
            return None

        try:
            # Execute candidate code in the globals dict
            exec(candidate_code, self.globals_dict)

            # Use existing logic to extract the function
            candidate_func = self.get_function_callable(skeleton)
            if not candidate_func:
                return None

            return verify_func(candidate_func)

        except Exception:
            return None
