"""
S3-based storage backend for ProllyTree nodes.
"""

import os
import pickle
import tomllib
from typing import Optional, Iterator, List

import boto3
from botocore.exceptions import ClientError

from ..node import Node
from .protocols import BlockStore, Remote


class S3BlockStore(BlockStore, Remote):
    """S3-based block storage that implements both BlockStore and Remote protocols."""

    def __init__(self, bucket: str, prefix: str, access_key: str,
                 secret_key: str, region: str = 'us-east-1'):
        """
        Initialize S3 block store.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all blobs
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region (default: us-east-1)
        """
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

    @classmethod
    def from_config(cls, config_path: str, remote_name: str = 'origin') -> 'S3BlockStore':
        """
        Create S3BlockStore from a TOML config file.

        Config format:
            [remotes.origin]
            type = "s3"
            bucket = "your-bucket-name"
            prefix = "prolly/"
            access_key_id = "YOUR_ACCESS_KEY"
            secret_access_key = "YOUR_SECRET_KEY"
            region = "us-east-1"

        Args:
            config_path: Path to TOML config file
            remote_name: Name of the remote to load (default: origin)

        Returns:
            S3BlockStore instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required config fields are missing
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, 'rb') as f:
            config = tomllib.load(f)

        remotes = config.get('remotes', {})
        if remote_name not in remotes:
            raise ValueError(f"Remote '{remote_name}' not found in config")

        remote_config = remotes[remote_name]
        remote_type = remote_config.get('type', 's3')
        if remote_type != 's3':
            raise ValueError(f"Remote '{remote_name}' is not an S3 remote (type: {remote_type})")

        bucket = remote_config.get('bucket')
        prefix = remote_config.get('prefix', '')
        access_key = remote_config.get('access_key_id')
        secret_key = remote_config.get('secret_access_key')
        region = remote_config.get('region', 'us-east-1')

        if not bucket:
            raise ValueError(f"'bucket' not specified for remote '{remote_name}'")
        if not access_key or not secret_key:
            raise ValueError(f"'access_key_id' and 'secret_access_key' required for remote '{remote_name}'")

        return cls(bucket, prefix, access_key, secret_key, region)

    def put_node(self, node_hash: bytes, node: Node):
        """Store a node to S3."""
        data = pickle.dumps(node)
        hash_hex = node_hash.hex()
        key = f"{self.prefix}blocks/{hash_hex[:2]}/{hash_hex[2:]}"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data
        )

    def get_node(self, node_hash: bytes) -> Optional[Node]:
        """Retrieve a node from S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.get_node not yet implemented")

    def url(self) -> str:
        """Return S3 URL for this store."""
        return f"s3://{self.bucket}/{self.prefix}"

    def delete_node(self, node_hash: bytes) -> bool:
        """Delete a node from S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.delete_node not yet implemented")

    def list_nodes(self) -> Iterator[bytes]:
        """List all nodes in S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.list_nodes not yet implemented")

    def count_nodes(self) -> int:
        """Count nodes in S3. Not yet implemented."""
        raise NotImplementedError("S3BlockStore.count_nodes not yet implemented")

    # Remote protocol methods

    def list_refs(self) -> List[str]:
        """List all refs on the remote."""
        refs = []
        prefix = f"{self.prefix}refs/"

        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                # Extract ref name from key like "prefix/refs/main"
                ref_name = obj['Key'][len(prefix):]
                refs.append(ref_name)

        return refs

    def get_ref_commit(self, ref_name: str) -> Optional[str]:
        """Get the commit hash for a ref. Returns None if ref doesn't exist."""
        key = f"{self.prefix}refs/{ref_name}"

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read().decode('utf-8').strip()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def update_ref(self, ref_name: str, old_hash: Optional[str], new_hash: str) -> bool:
        """
        Update a ref with CAS semantics using S3 conditional writes.

        Uses ETag for conditional updates to detect concurrent modifications.

        Returns True if update succeeded, False if there was a conflict.
        """
        key = f"{self.prefix}refs/{ref_name}"

        try:
            if old_hash is None:
                # Creating new ref - should not exist
                # Use IfNoneMatch to ensure it doesn't exist
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=new_hash.encode('utf-8'),
                    IfNoneMatch='*'
                )
            else:
                # Updating existing ref - get current ETag first
                try:
                    response = self.s3.get_object(Bucket=self.bucket, Key=key)
                    current_value = response['Body'].read().decode('utf-8').strip()
                    etag = response['ETag']

                    # Check if current value matches expected old_hash
                    if current_value != old_hash:
                        return False

                    # Update with ETag condition
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=new_hash.encode('utf-8'),
                        IfMatch=etag
                    )
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        # Ref doesn't exist but we expected it to
                        return False
                    raise

            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            # PreconditionFailed means ETag didn't match (concurrent update)
            # or object already exists (for IfNoneMatch)
            if error_code in ('PreconditionFailed', 'ConditionalRequestConflict'):
                return False
            raise
