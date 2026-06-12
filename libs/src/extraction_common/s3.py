import os

import s3fs
from dotenv import load_dotenv

load_dotenv()


def get_s3_fs() -> s3fs.S3FileSystem:
    endpoint = os.getenv("AWS_S3_ENDPOINT")
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return s3fs.S3FileSystem(
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        token=os.getenv("AWS_SESSION_TOKEN"),
        client_kwargs={
            "endpoint_url": endpoint,
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
        },
    )
