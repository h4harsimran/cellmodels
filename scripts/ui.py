#!/usr/bin/env python3
"""
CLI entry point to start the cellmodels web UI server.
"""

import argparse
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path to allow module loading
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

def main():
    parser = argparse.ArgumentParser(description="Start the cellmodels interactive UI dashboard")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    print(f"Starting cellmodels UI server at http://{args.host}:{args.port}")
    uvicorn.run(
        "cellmodels.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
