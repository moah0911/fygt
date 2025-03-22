#!/usr/bin/env python
"""
Run script for the Edumate application
"""

import os
import sys
from subprocess import Popen

def main():
    """Run the Edumate Streamlit application"""
    print("Starting Edumate Educational Platform...")
    
    # Get the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Add the current directory to the Python path
    sys.path.insert(0, current_dir)
    
    # Run the Streamlit application
    cmd = ["streamlit", "run", os.path.join(current_dir, "edumate", "app.py")]
    print(f"Running command: {' '.join(cmd)}")
    
    # Start the Streamlit server
    process = Popen(cmd)
    
    try:
        # Wait for the process to complete
        process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nShutting down Edumate...")
        process.terminate()
        process.wait()
        print("Edumate has been shut down.")

if __name__ == "__main__":
    main() 