//! S3-based storage backend for ProllyTree nodes.

use aws_sdk_s3::Client;
use aws_sdk_s3::config::{Credentials, Region};
use prolly_core::{BlockStore, Commit, Hash, Node, Remote, StoreError, StoreResult};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// S3 remote configuration from TOML config file.
#[derive(Debug, Deserialize)]
pub struct S3Config {
    pub bucket: String,
    pub prefix: Option<String>,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RemoteConfig {
    #[serde(rename = "type")]
    remote_type: Option<String>,
    #[serde(flatten)]
    s3: S3Config,
}

#[derive(Debug, Deserialize)]
struct Config {
    remotes: HashMap<String, RemoteConfig>,
}

/// S3-based block storage that implements both BlockStore and Remote traits.
pub struct S3BlockStore {
    client: Client,
    bucket: String,
    prefix: String,
    runtime: tokio::runtime::Runtime,
    // Local cache for nodes to avoid repeated S3 calls
    node_cache: Mutex<HashMap<Hash, Arc<Node>>>,
    commit_cache: Mutex<HashMap<Hash, Commit>>,
}

impl S3BlockStore {
    /// Create a new S3 block store.
    pub fn new(
        bucket: String,
        prefix: String,
        access_key: String,
        secret_key: String,
        region: String,
    ) -> io::Result<Self> {
        // Create tokio runtime for async S3 operations
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Create S3 client
        let client = runtime.block_on(async {
            let credentials = Credentials::new(
                access_key,
                secret_key,
                None,
                None,
                "prolly-cli",
            );

            let config = aws_sdk_s3::Config::builder()
                .region(Region::new(region))
                .credentials_provider(credentials)
                .behavior_version_latest()
                .build();

            Client::from_conf(config)
        });

        Ok(S3BlockStore {
            client,
            bucket,
            prefix,
            runtime,
            node_cache: Mutex::new(HashMap::new()),
            commit_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Create S3BlockStore from a TOML config file.
    pub fn from_config(config_path: &Path, remote_name: &str) -> io::Result<Self> {
        if !config_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Config not found at {}", config_path.display()),
            ));
        }

        let config_str = fs::read_to_string(config_path)?;
        let config: Config = toml::from_str(&config_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let remote_config = config.remotes.get(remote_name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Remote '{}' not found in config", remote_name),
            )
        })?;

        // Verify remote type is S3
        if let Some(ref t) = remote_config.remote_type {
            if t != "s3" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Remote '{}' is not an S3 remote (type: {})", remote_name, t),
                ));
            }
        }

        let s3_config = &remote_config.s3;
        Self::new(
            s3_config.bucket.clone(),
            s3_config.prefix.clone().unwrap_or_default(),
            s3_config.access_key_id.clone(),
            s3_config.secret_access_key.clone(),
            s3_config.region.clone().unwrap_or_else(|| "us-east-1".to_string()),
        )
    }

    /// Get the S3 key for a node hash.
    fn node_key(&self, node_hash: &Hash) -> String {
        let hash_hex = hex::encode(node_hash);
        format!("{}blocks/{}/{}", self.prefix, &hash_hex[..2], &hash_hex[2..])
    }

    /// Get the S3 key for a commit hash.
    fn commit_key(&self, commit_hash: &Hash) -> String {
        format!("{}commits/{}", self.prefix, hex::encode(commit_hash))
    }

    /// Get the S3 key for a ref.
    fn ref_key(&self, ref_name: &str) -> String {
        format!("{}refs/{}", self.prefix, ref_name)
    }
}

// Mark S3BlockStore as Send + Sync
unsafe impl Send for S3BlockStore {}
unsafe impl Sync for S3BlockStore {}

impl BlockStore for S3BlockStore {
    fn put_node(&self, node_hash: &Hash, node: Node) -> StoreResult<()> {
        let key = self.node_key(node_hash);
        let data = bincode::serialize(&node)?;

        let client = self.client.clone();
        let bucket = self.bucket.clone();

        self.runtime.block_on(async {
            client
                .put_object()
                .bucket(&bucket)
                .key(&key)
                .body(data.into())
                .send()
                .await
                .map_err(|e| StoreError::Network(format!("S3 put_object failed: {}", e)))
        })?;

        // Update cache
        let mut cache = self.node_cache.lock()?;
        cache.insert(node_hash.clone(), Arc::new(node));
        Ok(())
    }

    fn get_node(&self, node_hash: &Hash) -> StoreResult<Option<Arc<Node>>> {
        // Check cache first
        {
            let cache = self.node_cache.lock()?;
            if let Some(node) = cache.get(node_hash) {
                return Ok(Some(node.clone()));
            }
        }

        let key = self.node_key(node_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let result = self.runtime.block_on(async {
            match client
                .get_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
            {
                Ok(response) => Ok(Some(response)),
                Err(e) => {
                    // Check if it's a "not found" error
                    let is_not_found = e.as_service_error()
                        .map(|se| se.is_no_such_key())
                        .unwrap_or(false);
                    if is_not_found {
                        Ok(None)
                    } else {
                        Err(StoreError::Network(format!("S3 get_object failed: {}", e)))
                    }
                }
            }
        })?;

        let response = match result {
            Some(r) => r,
            None => return Ok(None),
        };

        let data = self.runtime.block_on(async {
            response.body.collect().await
                .map(|b| b.to_vec())
                .map_err(|e| StoreError::Network(format!("S3 read body failed: {}", e)))
        })?;

        let node: Node = bincode::deserialize(&data)
            .map_err(|e| StoreError::Deserialization(format!("Failed to deserialize node: {}", e)))?;

        let node_arc: Arc<Node> = Arc::new(node);

        // Update cache
        let mut cache = self.node_cache.lock()?;
        cache.insert(node_hash.clone(), node_arc.clone());
        Ok(Some(node_arc))
    }

    fn delete_node(&self, node_hash: &Hash) -> StoreResult<bool> {
        let key = self.node_key(node_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        self.runtime.block_on(async {
            client
                .delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
                .map_err(|e| StoreError::Network(format!("S3 delete_object failed: {}", e)))
        })?;

        // Remove from cache
        let mut cache = self.node_cache.lock()?;
        cache.remove(node_hash);

        Ok(true)
    }

    fn list_nodes(&self) -> StoreResult<Vec<Hash>> {
        let prefix = format!("{}blocks/", self.prefix);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let nodes = self.runtime.block_on(async {
            let mut nodes = Vec::new();
            let mut continuation_token = None;

            loop {
                let mut request = client
                    .list_objects_v2()
                    .bucket(&bucket)
                    .prefix(&prefix);

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                let response = request.send().await
                    .map_err(|e| StoreError::Network(format!("S3 list_objects failed: {}", e)))?;

                for object in response.contents() {
                    if let Some(key) = object.key() {
                        // Extract hash from key like "prefix/blocks/ab/cdef..."
                        let hash_part = key.strip_prefix(&prefix);
                        if let Some(hash_str) = hash_part {
                            // Remove the "/" between first two chars and rest
                            let cleaned: String = hash_str.chars().filter(|c| *c != '/').collect();
                            if let Ok(hash) = hex::decode(&cleaned) {
                                nodes.push(hash);
                            }
                        }
                    }
                }

                if response.is_truncated() == Some(true) {
                    continuation_token = response.next_continuation_token().map(String::from);
                } else {
                    break;
                }
            }
            Ok::<_, StoreError>(nodes)
        })?;

        Ok(nodes)
    }

    fn count_nodes(&self) -> StoreResult<usize> {
        Ok(self.list_nodes()?.len())
    }
}

impl Remote for S3BlockStore {
    fn url(&self) -> String {
        format!("s3://{}/{}", self.bucket, self.prefix)
    }

    fn list_refs(&self) -> StoreResult<Vec<String>> {
        let prefix = format!("{}refs/", self.prefix);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let refs = self.runtime.block_on(async {
            let mut refs = Vec::new();
            let mut continuation_token = None;

            loop {
                let mut request = client
                    .list_objects_v2()
                    .bucket(&bucket)
                    .prefix(&prefix);

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                let response = request.send().await
                    .map_err(|e| StoreError::Network(format!("S3 list_objects failed: {}", e)))?;

                for object in response.contents() {
                    if let Some(key) = object.key() {
                        if let Some(ref_name) = key.strip_prefix(&prefix) {
                            refs.push(ref_name.to_string());
                        }
                    }
                }

                if response.is_truncated() == Some(true) {
                    continuation_token = response.next_continuation_token().map(String::from);
                } else {
                    break;
                }
            }
            Ok::<_, StoreError>(refs)
        })?;

        Ok(refs)
    }

    fn get_ref_commit(&self, ref_name: &str) -> StoreResult<Option<String>> {
        let key = self.ref_key(ref_name);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        self.runtime.block_on(async {
            match client.get_object().bucket(&bucket).key(&key).send().await {
                Ok(response) => {
                    let data = response.body.collect().await
                        .map_err(|e| StoreError::Network(format!("S3 read body failed: {}", e)))?;
                    let s = String::from_utf8(data.to_vec())
                        .map_err(|e| StoreError::Deserialization(format!("Invalid UTF-8: {}", e)))?;
                    Ok(Some(s.trim().to_string()))
                }
                Err(e) => {
                    let is_not_found = e.as_service_error()
                        .map(|se| se.is_no_such_key())
                        .unwrap_or(false);
                    if is_not_found {
                        Ok(None)
                    } else {
                        Err(StoreError::Network(format!("S3 get_object failed: {}", e)))
                    }
                }
            }
        })
    }

    fn update_ref(&self, ref_name: &str, old_hash: Option<&str>, new_hash: &str) -> StoreResult<()> {
        let key = self.ref_key(ref_name);
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let new_hash = new_hash.to_string();
        let ref_name_owned = ref_name.to_string();
        let old_hash_owned = old_hash.map(|s| s.to_string());

        self.runtime.block_on(async {
            if old_hash_owned.is_none() {
                // Creating new ref - should not exist
                // Use if_none_match("*") to ensure it doesn't exist (atomic)
                let result = client
                    .put_object()
                    .bucket(&bucket)
                    .key(&key)
                    .body(new_hash.into_bytes().into())
                    .if_none_match("*")
                    .send()
                    .await;

                match result {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        // Check if it's a precondition failed (object already exists)
                        let is_precondition_failed = e.as_service_error()
                            .map(|se| {
                                // S3 returns PreconditionFailed when if_none_match fails
                                se.meta().code() == Some("PreconditionFailed")
                            })
                            .unwrap_or(false);
                        if is_precondition_failed {
                            // Get the actual value for the error message
                            let actual = match client.get_object().bucket(&bucket).key(&key).send().await {
                                Ok(response) => {
                                    let data = response.body.collect().await.ok();
                                    data.and_then(|d| String::from_utf8(d.to_vec()).ok())
                                        .map(|s| s.trim().to_string())
                                }
                                Err(_) => Some("<unknown>".to_string()),
                            };
                            Err(StoreError::RefConflict {
                                ref_name: ref_name_owned,
                                expected: None,
                                actual,
                            })
                        } else {
                            Err(StoreError::Network(format!("S3 put_object failed: {}", e)))
                        }
                    }
                }
            } else {
                // Updating existing ref - get current value and ETag first
                let (current_value, etag) = match client.get_object().bucket(&bucket).key(&key).send().await {
                    Ok(response) => {
                        let etag = response.e_tag().map(|s| s.to_string());
                        let data = response.body.collect().await
                            .map_err(|e| StoreError::Network(format!("S3 read body failed: {}", e)))?;
                        let s = String::from_utf8(data.to_vec())
                            .map_err(|e| StoreError::Deserialization(format!("Invalid UTF-8: {}", e)))?;
                        (Some(s.trim().to_string()), etag)
                    }
                    Err(e) => {
                        let is_not_found = e.as_service_error()
                            .map(|se| se.is_no_such_key())
                            .unwrap_or(false);
                        if is_not_found {
                            (None, None)
                        } else {
                            return Err(StoreError::Network(format!("S3 get_object failed: {}", e)));
                        }
                    }
                };

                // Check if current value matches expected old_hash
                let expected = old_hash_owned.as_ref();
                match (&current_value, expected) {
                    (Some(actual), Some(exp)) if actual != exp => {
                        return Err(StoreError::RefConflict {
                            ref_name: ref_name_owned,
                            expected: old_hash_owned,
                            actual: current_value,
                        });
                    }
                    (None, Some(_)) => {
                        return Err(StoreError::RefConflict {
                            ref_name: ref_name_owned,
                            expected: old_hash_owned,
                            actual: None,
                        });
                    }
                    _ => {} // Matches, proceed
                }

                // Update with ETag condition for atomicity
                let etag = etag.ok_or_else(|| {
                    StoreError::Network("S3 get_object did not return ETag".to_string())
                })?;

                let result = client
                    .put_object()
                    .bucket(&bucket)
                    .key(&key)
                    .body(new_hash.into_bytes().into())
                    .if_match(&etag)
                    .send()
                    .await;

                match result {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        // Check if it's a precondition failed (ETag changed - concurrent update)
                        let is_precondition_failed = e.as_service_error()
                            .map(|se| {
                                se.meta().code() == Some("PreconditionFailed")
                            })
                            .unwrap_or(false);
                        if is_precondition_failed {
                            // Get the actual current value for the error message
                            let actual = match client.get_object().bucket(&bucket).key(&key).send().await {
                                Ok(response) => {
                                    let data = response.body.collect().await.ok();
                                    data.and_then(|d| String::from_utf8(d.to_vec()).ok())
                                        .map(|s| s.trim().to_string())
                                }
                                Err(_) => Some("<unknown>".to_string()),
                            };
                            Err(StoreError::RefConflict {
                                ref_name: ref_name_owned,
                                expected: old_hash_owned,
                                actual,
                            })
                        } else {
                            Err(StoreError::Network(format!("S3 put_object failed: {}", e)))
                        }
                    }
                }
            }
        })
    }

    fn put_commit(&self, commit_hash: &Hash, commit: Commit) -> StoreResult<()> {
        let key = self.commit_key(commit_hash);
        let data = serde_json::to_vec(&commit)?;
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        self.runtime.block_on(async {
            client
                .put_object()
                .bucket(&bucket)
                .key(&key)
                .body(data.into())
                .send()
                .await
                .map_err(|e| StoreError::Network(format!("S3 put_object failed: {}", e)))
        })?;

        // Update cache
        let mut cache = self.commit_cache.lock()?;
        cache.insert(commit_hash.clone(), commit);
        Ok(())
    }

    fn get_commit(&self, commit_hash: &Hash) -> StoreResult<Option<Commit>> {
        // Check cache first
        {
            let cache = self.commit_cache.lock()?;
            if let Some(commit) = cache.get(commit_hash) {
                return Ok(Some(commit.clone()));
            }
        }

        let key = self.commit_key(commit_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let result = self.runtime.block_on(async {
            match client.get_object().bucket(&bucket).key(&key).send().await {
                Ok(response) => {
                    let data = response.body.collect().await
                        .map_err(|e| StoreError::Network(format!("S3 read body failed: {}", e)))?;
                    let commit: Commit = serde_json::from_slice(&data.to_vec())
                        .map_err(|e| StoreError::Deserialization(format!("Failed to deserialize commit: {}", e)))?;
                    Ok(Some(commit))
                }
                Err(e) => {
                    let is_not_found = e.as_service_error()
                        .map(|se| se.is_no_such_key())
                        .unwrap_or(false);
                    if is_not_found {
                        Ok(None)
                    } else {
                        Err(StoreError::Network(format!("S3 get_object failed: {}", e)))
                    }
                }
            }
        })?;

        if let Some(ref commit) = result {
            // Update cache
            let mut cache = self.commit_cache.lock()?;
            cache.insert(commit_hash.clone(), commit.clone());
        }

        Ok(result)
    }

    fn get_parents(&self, commit_hash: &Hash) -> StoreResult<Vec<Hash>> {
        Ok(self.get_commit(commit_hash)?
            .map(|c| c.parents)
            .unwrap_or_default())
    }

    fn set_ref(&self, name: &str, commit_hash: &Hash) -> StoreResult<()> {
        let key = self.ref_key(name);
        let hex_hash = hex::encode(commit_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        self.runtime.block_on(async {
            client
                .put_object()
                .bucket(&bucket)
                .key(&key)
                .body(hex_hash.into_bytes().into())
                .send()
                .await
                .map_err(|e| StoreError::Network(format!("S3 put_object failed: {}", e)))
        })?;

        Ok(())
    }

    fn get_ref(&self, name: &str) -> StoreResult<Option<Hash>> {
        match self.get_ref_commit(name)? {
            Some(hex) => Ok(Some(hex::decode(&hex)?)),
            None => Ok(None),
        }
    }
}
