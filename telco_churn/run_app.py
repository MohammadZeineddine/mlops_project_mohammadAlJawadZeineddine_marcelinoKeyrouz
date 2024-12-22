import argparse
import os
from dotenv import load_dotenv
import uvicorn

def main():
    # Parse the command-line argument for the environment
    parser = argparse.ArgumentParser(description="FastAPI application for churn prediction.")
    parser.add_argument("--env", type=str, choices=["dev", "prod"], required=True, help="Set environment (dev or prod).")
    args = parser.parse_args()

    # Load the corresponding .env file based on the environment argument
    env_file = f".env.{args.env}"
    load_dotenv(dotenv_path=env_file)

    # Now, run the FastAPI app
    uvicorn.run("telco_churn.api:app", host="0.0.0.0", port=8000, reload=True)

# Add the `if __name__ == '__main__':` block to ensure proper handling of multiprocessing in Windows
if __name__ == "__main__":
    main()
