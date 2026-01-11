import os
import json
import uuid
import base64
import boto3
from datetime import datetime, timezone
from botocore.config import Config

# --- Fixed regions (your setup) ---
BEDROCK_REGION = "us-west-2"  # SD 3.5 Large
S3_REGION = "us-east-2"       # your bucket region

# --- Clients ---
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# Force SigV4 so the presigned URL always works
s3 = boto3.client(
    "s3",
    region_name=S3_REGION,
    config=Config(signature_version="s3v4")
)

# --- Config ---
MODEL_ID = os.environ.get("MODEL_ID", "stability.sd3-5-large-v1:0").strip()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "myovieostermageenerator03").strip()
KEY_PREFIX = os.environ.get("KEY_PREFIX", "generated/").strip()
URL_EXPIRES_SECONDS = int(os.environ.get("URL_EXPIRES_SECONDS", "3600"))


def _headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
    }


def _resp(status_code: int, payload: dict):
    return {
        "statusCode": status_code,
        "headers": _headers(),
        "body": json.dumps(payload),
    }


def _qsp(event: dict) -> dict:
    return (event or {}).get("queryStringParameters") or {}


def _get_param(event: dict, name: str, default: str = "") -> str:
    # Supports Lambda test event: {"prompt":"..."}
    if isinstance(event, dict) and event.get(name) is not None:
        return str(event.get(name))
    # Supports API Gateway proxy GET: ?prompt=...
    q = _qsp(event)
    if q.get(name) is not None:
        return str(q.get(name))
    return default


def lambda_handler(event, context):
    # CORS preflight
    if (event or {}).get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": _headers(), "body": json.dumps({"ok": True})}

    prompt = _get_param(event, "prompt", "").strip()
    if not prompt:
        return _resp(400, {"error": "Missing required parameter: prompt"})

    negative_prompt = _get_param(event, "negative_prompt", "").strip()
    aspect_ratio = _get_param(event, "aspect_ratio", "1:1").strip()
    output_format = _get_param(event, "output_format", "png").strip()

    # SD3.5 allowed fields (per the error message you saw earlier)
    request_body = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "mode": "text-to-image",
        "seed": 0,
        "output_format": output_format,
        "aspect_ratio": aspect_ratio,
    }

    br = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body),
    )

    data = json.loads(br["body"].read())

    # Extract base64 image (handle both possible shapes)
    if "images" in data and data["images"]:
        image_b64 = data["images"][0]
    elif "artifacts" in data and data["artifacts"]:
        image_b64 = data["artifacts"][0].get("base64")
    else:
        return _resp(502, {"error": "No image in Bedrock response", "raw": data})

    image_bytes = base64.b64decode(image_b64)

    # Upload to S3
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = KEY_PREFIX if KEY_PREFIX.endswith("/") else f"{KEY_PREFIX}/"
    key = f"{prefix}{ts}-{uuid.uuid4().hex}.{output_format}"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=image_bytes,
        ContentType=f"image/{output_format}",
    )

    # ✅ THIS is the URL you must use (pre-signed, includes ?X-Amz-...)
    presigned_url = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": BUCKET_NAME, "Key": key},
    ExpiresIn=URL_EXPIRES_SECONDS,
    HttpMethod="GET"
)

    # Optional: a plain object URL (will be AccessDenied if object is private)
    object_url = f"https://{BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{key}"

    # Debug: confirm presigned URL really includes X-Amz-Algorithm
    print("PRESIGNED_HAS_X_AMZ:", "X-Amz-Algorithm=" in presigned_url)
    print("FULL_PRESIGNED_URL:", presigned_url)

    return {
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,OPTIONS"
    },
    "body": json.dumps({
        "presigned_url": presigned_url
    })
}

