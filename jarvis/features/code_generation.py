"""
RILEY - Code Generation System

This module enables RILEY to generate, analyze, and optimize code.
It provides the ability to:
1. Generate new Python modules and functions
2. Refactor and optimize existing code
3. Analyze codebases for improvement opportunities
4. Learn from code patterns and best practices
"""

import os
import sys
import re
import ast
import logging
import inspect
import importlib
import textwrap
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Configure logging
logger = logging.getLogger(__name__)

class CodeGenerator:
    """A system for generating and improving code."""
    
    def __init__(self, base_path: str = None):
        """Initialize the code generator.
        
        Args:
            base_path: Base path of the RILEY project. If None, will attempt to detect.
        """
        self.base_path = base_path or self._detect_base_path()
        
        # Define code quality standards and patterns
        self.standards = {
            "max_line_length": 100,
            "docstring_required": True,
            "type_hints_preferred": True,
            "prefer_f_strings": True
        }
        
        # Track ongoing code tasks
        self.ongoing_improvements = []
        
        logger.info("Code generator initialized")
    
    def _detect_base_path(self) -> str:
        """Auto-detect the base path of the RILEY project."""
        # Get the current file's directory and navigate up to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from features to the jarvis directory
        jarvis_dir = os.path.dirname(current_dir)
        # Go up one more level to the project root
        base_dir = os.path.dirname(jarvis_dir)
        return base_dir
    
    def generate_function(self, 
                         function_name: str, 
                         purpose: str,
                         params: List[Dict[str, str]] = None, 
                         return_type: str = None,
                         code_logic: str = None) -> str:
        """Generate a Python function with proper documentation.
        
        Args:
            function_name: Name of the function
            purpose: Description of what the function does
            params: List of parameter dictionaries with keys 'name', 'type', and 'description'
            return_type: Return type of the function
            code_logic: The actual implementation code (without def line or docstring)
            
        Returns:
            Properly formatted function code as a string
        """
        # Initialize with function definition
        if params is None:
            params = []
            
        # Create the function signature
        param_strings = []
        for param in params:
            param_str = param['name']
            if self.standards["type_hints_preferred"] and 'type' in param:
                param_str += f": {param['type']}"
            if 'default' in param:
                param_str += f" = {param['default']}"
            param_strings.append(param_str)
            
        # Add return type annotation if provided and type hints are preferred
        return_annotation = ""
        if return_type and self.standards["type_hints_preferred"]:
            return_annotation = f" -> {return_type}"
            
        # Start building the function
        function_def = f"def {function_name}({', '.join(param_strings)}){return_annotation}:"
        
        # Create docstring
        docstring = [f'"""{purpose}', ""]
        if params:
            docstring.append("Args:")
            for param in params:
                docstring.append(f"    {param['name']}: {param.get('description', 'Parameter description')}")
            docstring.append("")
        
        if return_type:
            docstring.append("Returns:")
            docstring.append(f"    {return_type}: Return value description")
            
        docstring.append('"""')
        
        # Process the code logic - add 4 spaces of indentation
        if code_logic:
            # Ensure the code is properly indented
            logic_lines = code_logic.strip().split('\n')
            indented_logic = ['    ' + line for line in logic_lines]
            
            # If no explicit return and we have a return type, add a default return
            if return_type and not any(line.strip().startswith('return ') for line in logic_lines):
                if return_type == 'bool':
                    indented_logic.append('    return False')
                elif return_type == 'int':
                    indented_logic.append('    return 0')
                elif return_type == 'float':
                    indented_logic.append('    return 0.0')
                elif return_type == 'str':
                    indented_logic.append('    return ""')
                elif return_type == 'List' or return_type.startswith('List['):
                    indented_logic.append('    return []')
                elif return_type == 'Dict' or return_type.startswith('Dict['):
                    indented_logic.append('    return {}')
                elif return_type != 'None':
                    indented_logic.append('    return None')
        else:
            # Default implementation
            if return_type == 'bool':
                indented_logic = ['    # TODO: Implement function logic', '    return False']
            elif return_type == 'int':
                indented_logic = ['    # TODO: Implement function logic', '    return 0']
            elif return_type == 'float':
                indented_logic = ['    # TODO: Implement function logic', '    return 0.0']
            elif return_type == 'str':
                indented_logic = ['    # TODO: Implement function logic', '    return ""']
            elif return_type == 'List' or (return_type and return_type.startswith('List[')):
                indented_logic = ['    # TODO: Implement function logic', '    return []']
            elif return_type == 'Dict' or (return_type and return_type.startswith('Dict[')):
                indented_logic = ['    # TODO: Implement function logic', '    return {}']
            elif return_type == 'None':
                indented_logic = ['    # TODO: Implement function logic', '    pass']
            else:
                indented_logic = ['    # TODO: Implement function logic', '    return None']
        
        # Assemble the full function
        full_function = [function_def]
        indented_docstring = ['    ' + line for line in docstring]
        full_function.extend(indented_docstring)
        full_function.extend(indented_logic)
        
        # Return as a string
        return '\n'.join(full_function)
    
    def generate_class(self, 
                      class_name: str, 
                      purpose: str,
                      attributes: List[Dict[str, str]] = None,
                      methods: List[Dict[str, Any]] = None,
                      base_classes: List[str] = None) -> str:
        """Generate a Python class with proper documentation.
        
        Args:
            class_name: Name of the class
            purpose: Description of what the class does
            attributes: List of attribute dictionaries with keys 'name', 'type', and 'description'
            methods: List of method dictionaries with method generation parameters
            base_classes: List of base classes to inherit from
            
        Returns:
            Properly formatted class code as a string
        """
        if attributes is None:
            attributes = []
        if methods is None:
            methods = []
        if base_classes is None:
            base_classes = []
            
        # Create the class definition line
        if base_classes:
            class_def = f"class {class_name}({', '.join(base_classes)}):"
        else:
            class_def = f"class {class_name}:"
            
        # Create docstring
        docstring = [f'"""{purpose}', ""]
        
        if attributes:
            docstring.append("Attributes:")
            for attr in attributes:
                docstring.append(f"    {attr['name']}: {attr.get('description', 'Attribute description')}")
            docstring.append("")
            
        docstring.append('"""')
        
        # Start building the class
        class_lines = [class_def]
        indented_docstring = ['    ' + line for line in docstring]
        class_lines.extend(indented_docstring)
        
        # Add attributes with default values if specified
        if attributes:
            class_lines.append("")  # Add a blank line after docstring
            for attr in attributes:
                if 'default' in attr:
                    class_lines.append(f"    {attr['name']} = {attr['default']}")
            
        # Generate methods
        if methods:
            # Always include __init__ method first if present
            init_method = next((m for m in methods if m.get('name') == '__init__'), None)
            if init_method:
                methods.remove(init_method)
                # Generate __init__ method
                init_code = self.generate_function(
                    function_name=init_method['name'],
                    purpose=init_method.get('purpose', 'Initialize the class.'),
                    params=init_method.get('params', []),
                    return_type=init_method.get('return_type', 'None'),
                    code_logic=init_method.get('code_logic', '')
                )
                # Indent the method code and add to class
                indented_init = textwrap.indent(init_code, '    ')
                class_lines.append("")  # Add a blank line before method
                class_lines.append(indented_init)
                
            # Generate all other methods
            for method in methods:
                method_code = self.generate_function(
                    function_name=method['name'],
                    purpose=method.get('purpose', f"The {method['name']} method."),
                    params=method.get('params', []),
                    return_type=method.get('return_type', None),
                    code_logic=method.get('code_logic', '')
                )
                # Indent the method code and add to class
                indented_method = textwrap.indent(method_code, '    ')
                class_lines.append("")  # Add a blank line between methods
                class_lines.append(indented_method)
                
        # If no methods were added, add a pass statement
        if not methods and not attributes:
            class_lines.append("    pass")
            
        # Return as a string
        return '\n'.join(class_lines)
    
    def generate_module(self,
                      module_name: str,
                      purpose: str,
                      imports: List[str] = None,
                      global_variables: List[Dict[str, Any]] = None,
                      functions: List[Dict[str, Any]] = None,
                      classes: List[Dict[str, Any]] = None) -> str:
        """Generate a complete Python module.
        
        Args:
            module_name: Name of the module
            purpose: Description of what the module does
            imports: List of import statements
            global_variables: List of global variable definitions
            functions: List of function dictionaries with function generation parameters
            classes: List of class dictionaries with class generation parameters
            
        Returns:
            Complete module code as a string
        """
        if imports is None:
            imports = []
        if global_variables is None:
            global_variables = []
        if functions is None:
            functions = []
        if classes is None:
            classes = []
            
        # Start with module docstring
        module_lines = [f'"""', f"RILEY - {module_name}", "", purpose, '"""', ""]
        
        # Add imports
        standard_imports = []
        third_party_imports = []
        riley_imports = []
        
        # Add some standard imports if not provided
        if imports:
            for imp in imports:
                if imp.startswith('import os') or imp.startswith('from os') or \
                   imp.startswith('import sys') or imp.startswith('from sys') or \
                   imp.startswith('import logging') or imp.startswith('from logging'):
                    standard_imports.append(imp)
                elif imp.startswith('import jarvis') or imp.startswith('from jarvis'):
                    riley_imports.append(imp)
                else:
                    third_party_imports.append(imp)
        
        # Add imports in the proper order
        if standard_imports:
            module_lines.extend(standard_imports)
            module_lines.append("")
        if third_party_imports:
            module_lines.extend(third_party_imports)
            module_lines.append("")
        if riley_imports:
            module_lines.extend(riley_imports)
            module_lines.append("")
            
        # Add a logging configuration line
        module_lines.append("# Configure logging")
        module_lines.append("logger = logging.getLogger(__name__)")
        module_lines.append("")
        
        # Add global variables
        if global_variables:
            for var in global_variables:
                if 'type' in var and self.standards["type_hints_preferred"]:
                    module_lines.append(f"{var['name']}: {var['type']} = {var['value']}")
                else:
                    module_lines.append(f"{var['name']} = {var['value']}")
            module_lines.append("")
            
        # Add classes
        for cls in classes:
            class_code = self.generate_class(
                class_name=cls['name'],
                purpose=cls.get('purpose', f"The {cls['name']} class."),
                attributes=cls.get('attributes', []),
                methods=cls.get('methods', []),
                base_classes=cls.get('base_classes', [])
            )
            module_lines.append(class_code)
            module_lines.append("")  # Blank line after class
            
        # Add functions
        for func in functions:
            function_code = self.generate_function(
                function_name=func['name'],
                purpose=func.get('purpose', f"The {func['name']} function."),
                params=func.get('params', []),
                return_type=func.get('return_type', None),
                code_logic=func.get('code_logic', '')
            )
            module_lines.append(function_code)
            module_lines.append("")  # Blank line after function
            
        # Return the complete module code
        return '\n'.join(module_lines)
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for quality issues and improvement opportunities.
        
        Args:
            code: The Python code to analyze
            
        Returns:
            Analysis results including issues and suggestions
        """
        results = {
            "issues": [],
            "suggestions": [],
            "metrics": {
                "lines_of_code": 0,
                "docstring_coverage": 0,
                "type_hint_coverage": 0
            }
        }
        
        # Count lines of code
        lines = code.split('\n')
        results["metrics"]["lines_of_code"] = len(lines)
        
        # Basic code style checks
        for i, line in enumerate(lines, 1):
            if len(line) > self.standards["max_line_length"]:
                results["issues"].append({
                    "line": i,
                    "issue": f"Line exceeds maximum length ({len(line)} > {self.standards['max_line_length']})",
                    "severity": "warning"
                })
                
            # Check for print statements (should use logging)
            if re.match(r'^\s*print\(', line):
                results["issues"].append({
                    "line": i,
                    "issue": "Using print() instead of logging",
                    "severity": "info",
                    "suggestion": "Replace print() with logger.info() or logger.debug()"
                })
        
        try:
            # Parse AST for more advanced analysis
            tree = ast.parse(code)
            
            # Count functions and classes
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            # Check docstring coverage
            functions_with_docstring = [f for f in functions if ast.get_docstring(f)]
            classes_with_docstring = [c for c in classes if ast.get_docstring(c)]
            
            if functions:
                function_docstring_ratio = len(functions_with_docstring) / len(functions)
                results["metrics"]["docstring_coverage"] = function_docstring_ratio
                
                if function_docstring_ratio < 1.0 and self.standards["docstring_required"]:
                    results["issues"].append({
                        "issue": f"Missing docstrings in {len(functions) - len(functions_with_docstring)} functions",
                        "severity": "warning"
                    })
            
            # Check for type hints
            type_hinted_functions = 0
            for func in functions:
                # Check return type annotation
                if func.returns:
                    type_hinted_functions += 1
                    continue
                    
                # Check argument type annotations
                if any(arg.annotation for arg in func.args.args):
                    type_hinted_functions += 1
                    
            if functions:
                type_hint_ratio = type_hinted_functions / len(functions)
                results["metrics"]["type_hint_coverage"] = type_hint_ratio
                
                if type_hint_ratio < 0.8 and self.standards["type_hints_preferred"]:
                    results["suggestions"].append({
                        "suggestion": "Add type hints to improve code quality and IDE support",
                        "priority": "medium"
                    })
                
        except SyntaxError as e:
            results["issues"].append({
                "line": e.lineno,
                "issue": f"Syntax error: {e.msg}",
                "severity": "error"
            })
        except Exception as e:
            results["issues"].append({
                "issue": f"Error analyzing code: {str(e)}",
                "severity": "error"
            })
            
        return results
    
    def suggest_code_improvements(self, module_path: str) -> List[Dict[str, Any]]:
        """Analyze a module and suggest improvements.
        
        Args:
            module_path: Path to the Python module to analyze
            
        Returns:
            List of suggested improvements
        """
        suggestions = []
        
        if not os.path.exists(module_path):
            logger.error(f"Module {module_path} does not exist")
            return suggestions
            
        try:
            # Read the module content
            with open(module_path, 'r') as f:
                code = f.read()
                
            # Analyze the code
            analysis = self.analyze_code(code)
            
            # Convert analysis to suggestions
            for issue in analysis["issues"]:
                if "suggestion" in issue:
                    suggestions.append({
                        "file": module_path,
                        "type": "fix_issue",
                        "description": issue["issue"],
                        "suggestion": issue["suggestion"],
                        "line": issue.get("line")
                    })
                    
            for suggestion in analysis["suggestions"]:
                suggestions.append({
                    "file": module_path,
                    "type": "enhancement",
                    "description": suggestion["suggestion"],
                    "priority": suggestion.get("priority", "medium")
                })
                
            # Add metrics-based suggestions
            metrics = analysis["metrics"]
            
            # Docstring coverage improvements
            if metrics["docstring_coverage"] < 1.0:
                suggestions.append({
                    "file": module_path,
                    "type": "documentation",
                    "description": f"Improve docstring coverage (currently {metrics['docstring_coverage']*100:.1f}%)",
                    "priority": "high" if metrics["docstring_coverage"] < 0.5 else "medium"
                })
                
            # Type hint improvements
            if metrics["type_hint_coverage"] < 0.8:
                suggestions.append({
                    "file": module_path,
                    "type": "code_quality",
                    "description": f"Add type hints (currently {metrics['type_hint_coverage']*100:.1f}%)",
                    "priority": "medium"
                })
                
        except Exception as e:
            logger.error(f"Error suggesting improvements for {module_path}: {e}")
            
        return suggestions
    
    def apply_improvement(self, improvement: Dict[str, Any]) -> Tuple[bool, str]:
        """Apply a code improvement.
        
        Args:
            improvement: Improvement details including file, description, and changes
            
        Returns:
            Tuple of (success, message)
        """
        file_path = improvement.get("file")
        if not file_path or not os.path.exists(file_path):
            return False, f"File {file_path} does not exist"
            
        try:
            with open(file_path, 'r') as f:
                original_code = f.read()
                
            new_code = original_code
            
            # Apply the changes
            change_type = improvement.get("type")
            
            if change_type == "code_replacement" and "old_code" in improvement and "new_code" in improvement:
                # Simple replacement
                if improvement["old_code"] in original_code:
                    new_code = original_code.replace(improvement["old_code"], improvement["new_code"])
                else:
                    return False, "Could not find the code to replace"
                    
            elif change_type == "add_function" and "function_code" in improvement:
                # Add a function to the module
                new_code = original_code + "\n\n" + improvement["function_code"]
                
            elif change_type == "add_method" and "class_name" in improvement and "method_code" in improvement:
                # Add a method to a class
                class_pattern = re.compile(f"class {improvement['class_name']}[:(]")
                class_match = class_pattern.search(original_code)
                
                if class_match:
                    # Find the end of the class
                    class_start = class_match.start()
                    lines = original_code.split('\n')
                    class_indent = 0
                    
                    for i, line in enumerate(lines):
                        if i == 0:
                            continue
                        if class_start > len('\n'.join(lines[:i])):
                            continue
                            
                        # Find first line of class definition
                        if "class " in line and class_match.group(0) in line:
                            # Get the indentation level
                            class_indent = len(line) - len(line.lstrip())
                            continue
                            
                        # Check if we're still in the class
                        if i < len(lines) - 1:
                            next_line = lines[i+1]
                            if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= class_indent:
                                # This is the end of the class
                                method_code = improvement["method_code"]
                                indented_method = textwrap.indent(method_code, ' ' * (class_indent + 4))
                                lines.insert(i+1, "")  # Add a blank line
                                lines.insert(i+2, indented_method)
                                new_code = '\n'.join(lines)
                                break
                    
                    if new_code == original_code:
                        # Couldn't find the end of the class, add to the end of the file
                        method_code = improvement["method_code"]
                        indented_method = textwrap.indent(method_code, ' ' * (class_indent + 4))
                        new_code = original_code + "\n\n" + indented_method
                        
                else:
                    return False, f"Could not find class {improvement['class_name']}"
            
            # Write the changes
            with open(file_path, 'w') as f:
                f.write(new_code)
                
            return True, f"Successfully applied improvement to {file_path}"
            
        except Exception as e:
            logger.error(f"Error applying improvement: {e}")
            return False, f"Error: {str(e)}"
    
    def generate_module_from_description(self, description: Dict[str, Any]) -> str:
        """Generate a complete module based on a high-level description.
        
        Args:
            description: Dictionary containing module details
            
        Returns:
            Generated module code
        """
        # Extract module info
        module_name = description.get("name", "unnamed_module")
        purpose = description.get("purpose", "A RILEY module.")
        
        # Basic imports needed
        imports = description.get("imports", [])
        standard_imports = ["import os", "import sys", "import logging"]
        for imp in standard_imports:
            if imp not in imports:
                imports.append(imp)
        
        # Generate the module
        return self.generate_module(
            module_name=module_name,
            purpose=purpose,
            imports=imports,
            global_variables=description.get("global_variables", []),
            functions=description.get("functions", []),
            classes=description.get("classes", [])
        )