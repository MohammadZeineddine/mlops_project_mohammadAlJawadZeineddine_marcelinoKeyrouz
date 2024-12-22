import argparse
from dotenv import load_dotenv
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="FastAPI application for churn prediction.")
    parser.add_argument("--env", type=str, choices=["dev", "prod"], required=True, help="Set environment (dev or prod).")
    args = parser.parse_args()

    env_file = f".env.{args.env}"
    load_dotenv(dotenv_path=env_file)

    uvicorn.run("telco_churn.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
