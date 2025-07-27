"""
Code Execution Tool for BabyAGI
Provides safe Python code execution capabilities.
"""

import time
import sys
import io
import contextlib
import ast
import traceback
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolResult, ToolStatus


class CodeExecutorTool(BaseTool):
    """Tool for executing Python code safely."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="code_executor",
            description="Execute Python code and return results",
            config=config
        )
        self.timeout = config.get("timeout", 30)
        self.max_output_length = config.get("max_output_length", 10000)
        
        # Restricted imports for security
        self.allowed_imports = {
            'math', 'random', 'datetime', 'json', 'csv', 'statistics',
            'collections', 'itertools', 'functools', 'operator',
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
            're', 'string', 'textwrap', 'unicodedata'
        }
        
        # Forbidden operations
        self.forbidden_patterns = [
            'import os', 'import sys', 'import subprocess', 'import socket',
            'import urllib', 'import requests', 'import http', 'import ftplib',
            'open(', 'file(', 'input(', 'raw_input(', 'eval(', 'exec(',
            '__import__', 'globals()', 'locals()', 'vars()', 'dir()',
            'getattr', 'setattr', 'delattr', 'hasattr'
        ]
    
    def execute(self, code: str, variables: Dict[str, Any] = None) -> ToolResult:
        """Execute Python code safely."""
        start_time = time.time()
        
        if not self.validate_parameters(code=code):
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message="Missing required parameter: code"
            )
        
        # Security checks
        security_check = self._check_code_security(code)
        if not security_check["safe"]:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Code security violation: {security_check['reason']}"
            )
        
        try:
            # Prepare execution environment
            exec_globals = self._prepare_execution_environment(variables or {})
            
            # Capture output
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer), \
                 contextlib.redirect_stderr(error_buffer):
                
                # Execute the code
                exec(code, exec_globals)
            
            # Get results
            stdout = output_buffer.getvalue()
            stderr = error_buffer.getvalue()
            
            # Limit output length
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... (output truncated)"
            
            execution_time = time.time() - start_time
            
            # Extract variables that were created/modified
            result_vars = {}
            for key, value in exec_globals.items():
                if not key.startswith('__') and key not in ['__builtins__']:
                    try:
                        # Only include serializable values
                        str(value)  # Test if it can be converted to string
                        result_vars[key] = value
                    except:
                        result_vars[key] = f"<{type(value).__name__} object>"
            
            self._update_stats(True)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "variables": result_vars,
                    "execution_successful": len(stderr) == 0
                },
                message="Code executed successfully" if len(stderr) == 0 else "Code executed with warnings",
                metadata={"code_length": len(code)},
                execution_time=execution_time
            )
            
        except Exception as e:
            self._update_stats(False)
            return ToolResult(
                status=ToolStatus.ERROR,
                data={
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                message=f"Code execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _check_code_security(self, code: str) -> Dict[str, Any]:
        """Check if code is safe to execute."""
        # Check for forbidden patterns
        code_lower = code.lower()
        for pattern in self.forbidden_patterns:
            if pattern in code_lower:
                return {"safe": False, "reason": f"Forbidden pattern detected: {pattern}"}
        
        # Parse AST to check for dangerous operations
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            return {"safe": False, "reason": f"Dangerous function call: {node.func.id}"}
                
                # Check for attribute access to dangerous modules
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in ['os', 'sys', 'subprocess']:
                            return {"safe": False, "reason": f"Access to dangerous module: {node.value.id}"}
        
        except SyntaxError as e:
            return {"safe": False, "reason": f"Syntax error: {str(e)}"}
        
        return {"safe": True, "reason": "Code passed security checks"}
    
    def _prepare_execution_environment(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a safe execution environment."""
        # Start with minimal builtins
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'format', 'frozenset', 'hex', 'int', 'len',
            'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'print',
            'range', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'zip'
        }
        
        # Create execution environment
        exec_globals = {
            '__builtins__': {name: getattr(__builtins__, name) for name in safe_builtins}
        }
        
        # Add allowed modules
        for module_name in self.allowed_imports:
            try:
                exec_globals[module_name] = __import__(module_name)
            except ImportError:
                pass  # Module not available, skip
        
        # Add user variables
        exec_globals.update(variables)
        
        return exec_globals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "required": ["code"],
            "optional": ["variables"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "variables": {
                    "type": "object",
                    "description": "Variables to make available in the execution environment"
                }
            }
        }

