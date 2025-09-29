import os
import json
import requests
from uuid import uuid4
from google.cloud import storage
from flask import Response
from typing import Optional

REPLIT_SIDECAR_ENDPOINT = "http://127.0.0.1:1106"

class ObjectNotFoundError(Exception):
    """Exception raised when an object is not found in storage."""
    pass

def create_storage_client():
    """Create and configure Google Cloud Storage client for Replit."""
    credentials_config = {
        "audience": "replit",
        "subject_token_type": "access_token",
        "token_url": f"{REPLIT_SIDECAR_ENDPOINT}/token",
        "type": "external_account",
        "credential_source": {
            "url": f"{REPLIT_SIDECAR_ENDPOINT}/credential",
            "format": {
                "type": "json",
                "subject_token_field_name": "access_token",
            },
        },
        "universe_domain": "googleapis.com",
    }
    
    return storage.Client(
        credentials=storage.Credentials.from_service_account_info(credentials_config),
        project=""
    )

class ObjectStorageService:
    """Service for managing object storage operations with Replit."""
    
    def __init__(self):
        self.client = create_storage_client()
    
    def get_private_object_dir(self) -> str:
        """Get the private object directory from environment."""
        dir_path = os.getenv('PRIVATE_OBJECT_DIR', '')
        if not dir_path:
            raise ValueError(
                "PRIVATE_OBJECT_DIR not set. Create a bucket in 'Object Storage' "
                "tool and set PRIVATE_OBJECT_DIR env var."
            )
        return dir_path
    
    def parse_object_path(self, path: str) -> tuple[str, str]:
        """Parse object path into bucket name and object name.
        
        Args:
            path: Path in format /<bucket_name>/<object_name>
            
        Returns:
            Tuple of (bucket_name, object_name)
        """
        if not path.startswith("/"):
            path = f"/{path}"
        
        parts = path.split("/")
        if len(parts) < 3:
            raise ValueError("Invalid path: must contain at least a bucket name")
        
        bucket_name = parts[1]
        object_name = "/".join(parts[2:])
        
        return bucket_name, object_name
    
    def get_upload_url(self) -> str:
        """Generate a presigned URL for uploading a new object.
        
        Returns:
            Presigned URL that can be used to upload a file
        """
        private_dir = self.get_private_object_dir()
        object_id = str(uuid4())
        full_path = f"{private_dir}/uploads/{object_id}"
        
        bucket_name, object_name = self.parse_object_path(full_path)
        
        return self.sign_object_url(
            bucket_name=bucket_name,
            object_name=object_name,
            method="PUT",
            ttl_sec=900  # 15 minutes
        )
    
    def sign_object_url(self, bucket_name: str, object_name: str, 
                       method: str, ttl_sec: int) -> str:
        """Sign an object URL using Replit sidecar.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            method: HTTP method (GET, PUT, DELETE, HEAD)
            ttl_sec: Time-to-live in seconds
            
        Returns:
            Signed URL string
        """
        import datetime
        
        expires_at = datetime.datetime.now() + datetime.timedelta(seconds=ttl_sec)
        
        request_data = {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "method": method,
            "expires_at": expires_at.isoformat()
        }
        
        response = requests.post(
            f"{REPLIT_SIDECAR_ENDPOINT}/object-storage/signed-object-url",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data)
        )
        
        if not response.ok:
            raise Exception(
                f"Failed to sign object URL, errorcode: {response.status_code}, "
                "make sure you're running on Replit"
            )
        
        return response.json()["signed_url"]
    
    def get_object_file(self, object_path: str):
        """Get a file object from storage.
        
        Args:
            object_path: Path in format /objects/<entity_id>
            
        Returns:
            Google Cloud Storage Blob object
        """
        if not object_path.startswith("/objects/"):
            raise ObjectNotFoundError()
        
        parts = object_path[1:].split("/")
        if len(parts) < 2:
            raise ObjectNotFoundError()
        
        entity_id = "/".join(parts[1:])
        entity_dir = self.get_private_object_dir()
        if not entity_dir.endswith("/"):
            entity_dir = f"{entity_dir}/"
        
        object_entity_path = f"{entity_dir}{entity_id}"
        bucket_name, object_name = self.parse_object_path(object_entity_path)
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        if not blob.exists():
            raise ObjectNotFoundError()
        
        return blob
    
    def download_object(self, blob, response: Response, cache_ttl_sec: int = 3600):
        """Stream an object to Flask response.
        
        Args:
            blob: Google Cloud Storage Blob object
            response: Flask response object
            cache_ttl_sec: Cache time-to-live in seconds
        """
        try:
            # Get blob metadata
            blob.reload()
            content_type = blob.content_type or "application/octet-stream"
            
            # Set response headers
            response.headers["Content-Type"] = content_type
            response.headers["Content-Length"] = str(blob.size)
            response.headers["Cache-Control"] = f"public, max-age={cache_ttl_sec}"
            
            # Stream file content
            response.data = blob.download_as_bytes()
            
        except Exception as e:
            print(f"Error downloading file: {e}")
            response.status_code = 500
            response.data = json.dumps({"error": "Error downloading file"})
    
    def normalize_object_entity_path(self, raw_path: str) -> str:
        """Normalize a raw object path to entity path format.
        
        Args:
            raw_path: Raw path (URL or path string)
            
        Returns:
            Normalized path in format /objects/<entity_id>
        """
        if not raw_path.startswith("https://storage.googleapis.com/"):
            return raw_path
        
        # Extract path from URL
        from urllib.parse import urlparse
        parsed = urlparse(raw_path)
        raw_object_path = parsed.path
        
        object_entity_dir = self.get_private_object_dir()
        if not object_entity_dir.endswith("/"):
            object_entity_dir = f"{object_entity_dir}/"
        
        if not raw_object_path.startswith(object_entity_dir):
            return raw_object_path
        
        # Extract entity ID
        entity_id = raw_object_path[len(object_entity_dir):]
        return f"/objects/{entity_id}"
    
    def upload_file_to_storage(self, file_data: bytes, filename: str, 
                               content_type: str) -> str:
        """Upload a file directly to storage.
        
        Args:
            file_data: File content as bytes
            filename: Name of the file
            content_type: MIME type of the file
            
        Returns:
            Object path in format /objects/<entity_id>
        """
        private_dir = self.get_private_object_dir()
        object_id = str(uuid4())
        full_path = f"{private_dir}/uploads/{object_id}"
        
        bucket_name, object_name = self.parse_object_path(full_path)
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Upload file
        blob.upload_from_string(file_data, content_type=content_type)
        
        # Return normalized path
        return f"/objects/uploads/{object_id}"
