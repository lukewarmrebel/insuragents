#!/usr/bin/env python
"""
Startup script for the Insurance LLM Framework.

This script sets up the environment and starts the Streamlit application.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Class responsible for setting up the application environment."""

    @staticmethod
    def create_directories(directories: List[Path]) -> None:
        """
        Create directories if they don't exist.

        Args:
            directories: List of directory paths to create
        """
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    @staticmethod
    def set_environment_variables() -> None:
        """Set necessary environment variables if not already set."""
        os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "1"
        logger.info("Disabled Streamlit watchdog")

    @staticmethod
    def check_anthropic_key() -> bool:
        """
        Check if Anthropic API key is set in environment.

        Returns:
            True if key is found, False otherwise
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            logger.info("Anthropic API key found")
            return True
        else:
            logger.warning("No Anthropic API key found in .env file")
            return False

    @classmethod
    def setup(cls) -> None:
        """Set up the complete environment for the application."""

        load_dotenv()
        logger.info("Loaded environment variables from .env file")

        required_dirs = [
            Path("evaluation/evaluations/forms"),
            Path("evaluation/evaluations/submissions"),
            Path("evaluation/benchmarks/results")
        ]
        cls.create_directories(required_dirs)

        cls.set_environment_variables()

        cls.check_anthropic_key()

        logger.info("Environment setup complete")

class ApplicationRunner:
    """Class responsible for running the Streamlit application."""

    @staticmethod
    def run(port: Optional[int] = None, host: Optional[str] = None) -> None:
        """
        Run the Streamlit application.

        Args:
            port: Port to run the application on
            host: Host to run the application on
        """

        port_str = str(port) if port else os.environ.get("APP_PORT", "8501")
        host_str = host if host else os.environ.get("APP_HOST", "0.0.0.0")

        cmd = [
            "streamlit", "run", "app.py",
            "--server.port", port_str,
            "--server.address", host_str
        ]

        logger.info(f"Starting application on {host_str}:{port_str}")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Error running application: {str(e)}")
            sys.exit(1)

class CommandLineParser:
    """Class for parsing command line arguments."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command line arguments.

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="Run the Insurance LLM Framework")
        parser.add_argument("--port", type=int,
                            help="Port to run the application on")
        parser.add_argument("--host", type=str,
                            help="Host to run the application on")
        return parser.parse_args()

def main() -> None:
    """Main function to set up and run the application."""

    args = CommandLineParser.parse_args()

    EnvironmentSetup.setup()

    ApplicationRunner.run(args.port, args.host)

if __name__ == "__main__":
    main()
