#!/usr/bin/env python3
"""
FastPaip Test Validator

Validates test files follow FastPaip testing conventions:
- Proper directory structure (unit/integration/e2e)
- Naming conventions (test_<action>_<condition>)
- No common anti-patterns
"""

import argparse
import ast
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationIssue:
    """Issue found during validation."""

    severity: str  # "error" or "warning"
    file: Path
    line: Optional[int]
    message: str


class TestValidator:
    """Validates test files against FastPaip conventions."""

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def validate_file(self, file_path: Path) -> None:
        """Validate a single test file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content, filename=str(file_path))

            # Check file naming
            self._check_file_naming(file_path)

            # Check structure
            self._check_structure(file_path, tree)

            # Check test functions
            self._check_test_functions(file_path, tree)

            # Check fixtures
            self._check_fixtures(file_path, tree)

            # Check imports
            self._check_imports(file_path, tree)

        except SyntaxError as e:
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    file=file_path,
                    line=e.lineno,
                    message=f"Syntax error: {e.msg}",
                )
            )

    def _check_file_naming(self, file_path: Path) -> None:
        """Check file follows naming convention."""
        if not file_path.name.startswith("test_"):
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    file=file_path,
                    line=None,
                    message=f"File must start with 'test_': {file_path.name}",
                )
            )

        if not file_path.name.endswith(".py"):
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    file=file_path,
                    line=None,
                    message=f"File must end with '.py': {file_path.name}",
                )
            )

    def _check_structure(self, file_path: Path, tree: ast.AST) -> None:
        """Check file is in correct directory structure."""
        parts = file_path.parts

        if "tests" not in parts:
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    file=file_path,
                    line=None,
                    message="Test file not in 'tests' directory",
                )
            )
            return

        tests_idx = parts.index("tests")
        if tests_idx + 1 >= len(parts):
            return

        test_type = parts[tests_idx + 1]
        valid_types = ["unit", "integration", "e2e"]

        if test_type not in valid_types:
            self.issues.append(
                ValidationIssue(
                    severity="warning",
                    file=file_path,
                    line=None,
                    message=f"Test not in standard directory. Expected: {', '.join(valid_types)}, got: {test_type}",
                )
            )

    def _check_test_functions(self, file_path: Path, tree: ast.AST) -> None:
        """Check test function conventions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    self._validate_test_function(file_path, node)

    def _validate_test_function(self, file_path: Path, node: ast.FunctionDef) -> None:
        """Validate a single test function."""
        # Check naming pattern
        if node.name == "test_":
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    file=file_path,
                    line=node.lineno,
                    message=f"Incomplete test name: {node.name}. Use: test_<action>_<condition>",
                )
            )

        # Check has docstring
        if not ast.get_docstring(node):
            self.issues.append(
                ValidationIssue(
                    severity="warning",
                    file=file_path,
                    line=node.lineno,
                    message=f"Test function missing docstring: {node.name}",
                )
            )

        # Check for anti-patterns
        self._check_for_anti_patterns(file_path, node)

    def _check_for_anti_patterns(self, file_path: Path, node: ast.FunctionDef) -> None:
        """Check for common testing anti-patterns."""
        for child in ast.walk(node):
            # Check for time.sleep (usually wrong in tests)
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if (
                        isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "time"
                        and child.func.attr == "sleep"
                    ):
                        self.issues.append(
                            ValidationIssue(
                                severity="warning",
                                file=file_path,
                                line=child.lineno,
                                message=f"Avoid time.sleep in tests (found in {node.name}). Use proper synchronization.",
                            )
                        )

            # Check for testing private methods
            if node.name.startswith("test__"):
                self.issues.append(
                    ValidationIssue(
                        severity="warning",
                        file=file_path,
                        line=node.lineno,
                        message=f"Testing private method: {node.name}. Test public behavior instead.",
                    )
                )

    def _check_fixtures(self, file_path: Path, tree: ast.AST) -> None:
        """Check fixture conventions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a fixture
                has_fixture_decorator = any(
                    isinstance(dec, ast.Name)
                    and dec.id == "fixture"
                    or isinstance(dec, ast.Attribute)
                    and dec.attr == "fixture"
                    for dec in node.decorator_list
                )

                if has_fixture_decorator:
                    self._validate_fixture(file_path, node)

    def _validate_fixture(self, file_path: Path, node: ast.FunctionDef) -> None:
        """Validate fixture naming and structure."""
        # Check mock fixture naming
        if "mock" in node.name.lower() and not node.name.startswith("mock_"):
            self.issues.append(
                ValidationIssue(
                    severity="warning",
                    file=file_path,
                    line=node.lineno,
                    message=f"Mock fixture should start with 'mock_': {node.name}",
                )
            )

    def _check_imports(self, file_path: Path, tree: ast.AST) -> None:
        """Check for problematic imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Warn about unittest.TestCase
                if node.module == "unittest" or node.module == "unittest.case":
                    if any(alias.name == "TestCase" for alias in node.names):
                        self.issues.append(
                            ValidationIssue(
                                severity="warning",
                                file=file_path,
                                line=node.lineno,
                                message="Avoid unittest.TestCase. Use pytest fixtures instead.",
                            )
                        )

    def validate_directory(self, dir_path: Path) -> None:
        """Validate all test files in directory."""
        for file_path in dir_path.rglob("test_*.py"):
            self.validate_file(file_path)

    def report(self) -> int:
        """Print validation report and return exit code."""
        if not self.issues:
            print("✅ All tests pass validation!")
            return 0

        # Group by severity
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]

        if errors:
            print(f"\n❌ {len(errors)} error(s) found:\n")
            for issue in errors:
                location = f"{issue.file}"
                if issue.line:
                    location += f":{issue.line}"
                print(f"  {location}")
                print(f"    {issue.message}\n")

        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s) found:\n")
            for issue in warnings:
                location = f"{issue.file}"
                if issue.line:
                    location += f":{issue.line}"
                print(f"  {location}")
                print(f"    {issue.message}\n")

        print(f"\nSummary: {len(errors)} errors, {len(warnings)} warnings")
        return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate test files follow FastPaip conventions"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to test file or directory to validate",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    validator = TestValidator()

    if args.path.is_file():
        validator.validate_file(args.path)
    else:
        validator.validate_directory(args.path)

    # Convert warnings to errors if strict mode
    if args.strict:
        for issue in validator.issues:
            if issue.severity == "warning":
                issue.severity = "error"

    exit_code = validator.report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
