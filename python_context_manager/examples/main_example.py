#!/usr/bin/env python3
"""
Main example runner for Python Context Manager.

This script provides a convenient way to run all examples or specific examples.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from examples import (
    run_basic_examples,
    run_advanced_examples,
    run_custom_tools_examples,
    run_performance_examples,
    run_all_examples,
    get_example_info
)


def main():
    """Main entry point for example runner."""
    parser = argparse.ArgumentParser(
        description="Python Context Manager Example Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all examples
  %(prog)s basic              # Run basic usage examples
  %(prog)s advanced           # Run advanced features examples
  %(prog)s tools              # Run custom tools examples
  %(prog)s performance        # Run performance demo examples
  %(prog)s info               # Show example information
        """
    )
    
    parser.add_argument(
        "example",
        nargs="?",
        choices=["all", "basic", "advanced", "tools", "performance", "info"],
        default="all",
        help="Which example to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.example == "info":
        show_example_info()
        return
    
    if args.verbose:
        print(f"Running example: {args.example}")
        print("-" * 50)
    
    try:
        if args.example == "all":
            run_all_examples()
        elif args.example == "basic":
            asyncio.run(run_basic_examples())
        elif args.example == "advanced":
            asyncio.run(run_advanced_examples())
        elif args.example == "tools":
            asyncio.run(run_custom_tools_examples())
        elif args.example == "performance":
            asyncio.run(run_performance_examples())
        
        if args.verbose:
            print("-" * 50)
            print(f"Example '{args.example}' completed successfully!")
            
    except KeyboardInterrupt:
        print("\nExample execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running example '{args.example}': {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def show_example_info():
    """Display information about available examples."""
    info = get_example_info()
    
    print("Python Context Manager Examples")
    print("=" * 40)
    print()
    
    for example_name, example_data in info.items():
        print(f"{example_name.replace('_', ' ').title():}")
        print(f"  Description: {example_data['description']}")
        print(f"  Topics: {', '.join(example_data['topics'])}")
        print(f"  Estimated time: {example_data['estimated_time']}")
        print()
    
    print("Usage:")
    print("  python main_example.py <example>")
    print()
    print("Available examples:")
    print("  all         - Run all examples")
    print("  basic       - Basic usage examples")
    print("  advanced    - Advanced features examples")
    print("  tools       - Custom tools examples")
    print("  performance - Performance demo examples")
    print("  info        - Show this information")


if __name__ == "__main__":
    main()