"""
Path utilities module for the Climbing Testing Application.

This module provides functions and utilities for managing file paths, directory
structures, and file naming conventions throughout the application. It ensures
consistent file management across different operating systems and environments.

Key functionalities:
- Generate standardized file paths for test data
- Create and manage directory structures
- Handle timestamp-based file naming
- Resolve relative paths in the application context

The path utilities ensure consistent file organization and access across
the application, regardless of the deployment environment.
"""
import os


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This is identified as the parent directory of the 'gui' directory.
    """
    # Start with this file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up until we find the project root (where 'gui' is a direct subdirectory)
    while os.path.basename(os.path.dirname(current_dir)) != 'gui' and os.path.dirname(current_dir) != current_dir:
        current_dir = os.path.dirname(current_dir)

    # Go up one more level to get to project root
    return os.path.dirname(os.path.dirname(current_dir))


def make_relative_path(absolute_path):
    """
    Converts an absolute path to a path relative to the test_page/tests directory.

    Parameters:
        absolute_path (str): The absolute path to convert

    Returns:
        str: A path relative to test_page/tests
    """
    if not absolute_path:
        return None

    # If it's already a relative path, return it as is
    if not os.path.isabs(absolute_path):
        return absolute_path

    # Find the 'tests' directory marker
    tests_marker = os.path.join('test_page', 'tests')

    if tests_marker in absolute_path:
        # Split the path at the 'tests' marker
        parts = absolute_path.split(tests_marker)
        # Return 'tests' + the rest of the path (properly joined)
        return os.path.join('tests', parts[1].lstrip(os.path.sep))

    # If no tests marker, try to make it relative to project root
    project_root = get_project_root()
    if absolute_path.startswith(project_root):
        return os.path.relpath(absolute_path, project_root)

    # If we can't make it relative, return the original path
    return absolute_path


def resolve_path(relative_path):
    """
    Converts a relative path or filename (stored in the database) to an absolute path.

    Parameters:
        relative_path (str): Path or filename of the test data

    Returns:
        str: The absolute path to the file
    """
    if not relative_path:
        return None

    # If it's already an absolute path, return it as is
    if os.path.isabs(relative_path):
        return relative_path

    # Get the script directory (where this utility file is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If the path starts with 'tests/', it's a relative path from test_page directory
    if relative_path.startswith('tests/'):
        return os.path.abspath(os.path.join(script_dir, relative_path))

    # If it's just a filename (no directory separators), assume it's in test_page/tests
    if '/' not in relative_path and '\\' not in relative_path:
        return os.path.abspath(os.path.join(script_dir, 'tests', relative_path))

    print(os.path.abspath(os.path.join(script_dir, relative_path)))
    # For other relative paths, try to resolve relative to gui/test_page directory
    return os.path.abspath(os.path.join(script_dir, relative_path))

# def resolve_path(relative_path):
#     """
#     Converts a relative path (stored in the database) to an absolute path.
#
#     Parameters:
#         relative_path (str): Filename of the test
#
#     Returns:
#         str: The absolute path
#     """
#     if not relative_path:
#         return None
#
#     # If it's already an absolute path, return it as is
#     if os.path.isabs(relative_path):
#         return relative_path
#
#     # # Get the base directory (test_page directory)
#     # script_dir = os.path.dirname(os.path.abspath(__file__))
#     #
#     # # If the path starts with 'tests/', resolve it relative to test_page directory
#     # if relative_path.startswith('tests/'):
#     #     return os.path.abspath(os.path.join(script_dir, relative_path))
#
#     # Otherwise, resolve relative to project root
#     project_root = get_project_root()
#     print(project_root)
#     abs_path = os.path.abspath(os.path.join(project_root, 'test_page', 'tests', relative_path))
#     print('ABSOLUT_PATH', abs_path)
#     return abs_path
