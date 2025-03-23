#!/usr/bin/env python
"""
Startup script to run both the main.py backend and the server.py frontend.
This launches both servers as separate processes.
"""

import os
import sys
import time
import subprocess
import signal
import argparse
import threading

# Define colors for terminal output
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Store processes for cleanup
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C and cleanup"""
    print(f"\n{TermColors.YELLOW}Shutting down servers...{TermColors.ENDC}")
    for process in processes:
        if process.poll() is None:  # If process is still running
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print(f"{TermColors.RED}Force killing process...{TermColors.ENDC}")
                process.kill()
    
    print(f"{TermColors.GREEN}All servers stopped.{TermColors.ENDC}")
    sys.exit(0)

def start_backend(port=8080):
    """Start the main.py backend API"""
    print(f"{TermColors.BLUE}Starting backend API on port {port}...{TermColors.ENDC}")
    
    # Set environment variables
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    # Start the process
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    processes.append(process)
    
    # Start a thread to monitor backend output
    threading.Thread(target=monitor_output, args=(process, "Backend"), daemon=True).start()
    
    return process

def start_frontend(port=5000, backend_port=8080):
    """Start the server.py frontend"""
    print(f"{TermColors.BLUE}Starting frontend server on port {port}...{TermColors.ENDC}")
    
    # Set environment variables
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    # Always set the backend API URL, defaulting to port 8080 if not specified
    if backend_port is None:
        backend_port = 8080
    
    env["BACKEND_API"] = f"http://localhost:{backend_port}"
    print(f"{TermColors.BLUE}Backend API at: http://localhost:{backend_port}{TermColors.ENDC}")
    
    # Start the process
    process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    processes.append(process)
    
    # Start a thread to monitor frontend output
    threading.Thread(target=monitor_output, args=(process, "Frontend"), daemon=True).start()
    
    return process

def wait_for_server(port, timeout=30):
    """Wait for a server to be ready"""
    import socket
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            socket.create_connection(("localhost", port), timeout=1)
            return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1)
    
    return False

def monitor_output(process, prefix):
    """Read and print output from a process"""
    for line in process.stdout:
        line = line.strip()
        # Print with color based on content
        if "ERROR" in line or "error" in line.lower():
            print(f"{TermColors.RED}{prefix}: {line}{TermColors.ENDC}")
        elif "WARNING" in line or "warning" in line.lower():
            print(f"{TermColors.YELLOW}{prefix}: {line}{TermColors.ENDC}")
        elif "DEBUG" in line or "debug" in line.lower():
            print(f"{TermColors.GREEN}{prefix}: {line}{TermColors.ENDC}")
        else:
            print(f"{TermColors.BOLD}{prefix}:{TermColors.ENDC} {line}")
        
        # Flush stdout to ensure logs appear immediately
        sys.stdout.flush()
    
    # Process has ended, check stderr
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"{TermColors.RED}{prefix} errors:{TermColors.ENDC}")
        print(stderr_output)
        sys.stdout.flush()
    
    # If we get here, the process has exited
    print(f"{TermColors.RED}{prefix} exited with code {process.returncode}{TermColors.ENDC}")
    sys.stdout.flush()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Podcast Guest Research application")
    parser.add_argument("--frontend-only", action="store_true", 
                        help="Start only the frontend server")
    parser.add_argument("--backend-port", type=int, default=8080,
                        help="Port for the backend API (default: 8080)")
    parser.add_argument("--frontend-port", type=int, default=5000,
                        help="Port for the frontend server (default: 5000)")
    return parser.parse_args()

def main():
    """Main function to start servers"""
    args = parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    backend_process = None
    if not args.frontend_only:
        try:
            backend_process = start_backend(args.backend_port)
            time.sleep(2)  # Give the backend a moment to start
            
            # Check if backend started successfully
            if backend_process.poll() is not None:
                print(f"{TermColors.RED}Backend failed to start. Running in frontend-only mode.{TermColors.ENDC}")
                print(f"{TermColors.YELLOW}Tip: Try running main.py separately:{TermColors.ENDC} python main.py")
                # Continue with frontend-only mode
            else:
                print(f"{TermColors.GREEN}Backend started successfully!{TermColors.ENDC}")
                
                # Wait for backend to be fully ready
                print(f"{TermColors.YELLOW}Waiting for backend to be ready...{TermColors.ENDC}")
                if wait_for_server(args.backend_port, timeout=10):
                    print(f"{TermColors.GREEN}Backend is ready to accept requests!{TermColors.ENDC}")
                else:
                    print(f"{TermColors.YELLOW}Backend did not respond to connection attempts, but process is running.{TermColors.ENDC}")
                    print(f"{TermColors.YELLOW}Continuing startup. Check backend logs if issues occur.{TermColors.ENDC}")
        except Exception as e:
            print(f"{TermColors.RED}Error starting backend: {str(e)}{TermColors.ENDC}")
            print(f"{TermColors.YELLOW}Running in frontend-only mode.{TermColors.ENDC}")
    
    try:
        frontend_process = start_frontend(
            args.frontend_port, 
            args.backend_port if backend_process and backend_process.poll() is None else None
        )
        
        time.sleep(2)  # Give the frontend a moment to start
        
        # Check if frontend started successfully
        if frontend_process.poll() is not None:
            print(f"{TermColors.RED}Frontend failed to start.{TermColors.ENDC}")
            signal_handler(None, None)  # Clean up any running processes
            return
    except Exception as e:
        print(f"{TermColors.RED}Error starting frontend: {str(e)}{TermColors.ENDC}")
        signal_handler(None, None)  # Clean up any running processes
        return
    
    # Success message
    if frontend_process.poll() is None:
        print(f"\n{TermColors.GREEN}Podcast Guest Research Web Interface is running!{TermColors.ENDC}")
        print(f"{TermColors.BOLD}Access the web interface at:{TermColors.ENDC} http://localhost:{args.frontend_port}")
        
        if backend_process and backend_process.poll() is None:
            print(f"{TermColors.BOLD}Backend API is running at:{TermColors.ENDC} http://localhost:{args.backend_port}")
        else:
            print(f"{TermColors.YELLOW}Note: Running in frontend-only mode. Start main.py separately if needed.{TermColors.ENDC}")
            
        print(f"{TermColors.YELLOW}Press Ctrl+C to stop all servers{TermColors.ENDC}")
    
    # Keep the main thread alive to handle signals
    try:
        # Just keep the main thread alive
        while any(p for p in processes if p.poll() is None):
            time.sleep(1)
            
        # If we get here, all processes have exited
        print(f"{TermColors.RED}All servers have stopped.{TermColors.ENDC}")
        
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main() 