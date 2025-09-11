#!/usr/bin/env python3
"""
Demo script showing how to use the global logger system.
This demonstrates that you don't need to setup logger in every file.
"""

from custom_logger import get_logger, setup_global_logger

def main():
    # Option 1: Setup global logger explicitly (recommended for main scripts)
    setup_global_logger()
    
    # Option 2: Just get logger (it will auto-setup if not already done)
    logger = get_logger()
    
    logger.info("This is from the main function")
    
    # Test different modules using the same logger
    test_module_a()
    test_module_b()

def test_module_a():
    # Each module can get its own child logger
    logger = get_logger("module_a")
    logger.info("This is from module A")
    logger.warning("This is a warning from module A")

def test_module_b():
    # Another module with its own child logger
    logger = get_logger("module_b")
    logger.info("This is from module B")
    logger.error("This is an error from module B")

if __name__ == "__main__":
    main()
