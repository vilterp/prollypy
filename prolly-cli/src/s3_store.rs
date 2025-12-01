//! S3-based storage backend for ProllyTree nodes.

use aws_sdk_s3::Client;
use aws_sdk_s3::config::{Credentials, Region};
use prolly_core::{BlockStore, Commit, Hash, Node, Remote};
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
    fn put_node(&self, node_hash: &Hash, node: Node) {
        let key = self.node_key(node_hash);
        let data = bincode::serialize(&node).expect("Failed to serialize node");

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
                .ok();
        });

        // Update cache
        let mut cache = self.node_cache.lock().unwrap();
        cache.insert(node_hash.clone(), Arc::new(node));
    }

    fn get_node(&self, node_hash: &Hash) -> Option<Arc<Node>> {
        // Check cache first
        {
            let cache = self.node_cache.lock().unwrap();
            if let Some(node) = cache.get(node_hash) {
                return Some(node.clone());
            }
        }

        let key = self.node_key(node_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let result = self.runtime.block_on(async {
            client
                .get_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
                .ok()
        });

        if let Some(response) = result {
            let data = self.runtime.block_on(async {
                response.body.collect().await.ok().map(|b| b.to_vec())
            });

            if let Some(data) = data {
                if let Ok(node) = bincode::deserialize::<Node>(&data) {
                    let node_arc: Arc<Node> = Arc::new(node);
                    // Update cache
                    let mut cache = self.node_cache.lock().unwrap();
                    cache.insert(node_hash.clone(), node_arc.clone());
                    return Some(node_arc);
                }
            }
        }

        None
    }

    fn delete_node(&self, node_hash: &Hash) -> bool {
        let key = self.node_key(node_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let result = self.runtime.block_on(async {
            client
                .delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
                .ok()
        });

        // Remove from cache
        let mut cache = self.node_cache.lock().unwrap();
        cache.remove(node_hash);

        result.is_some()
    }

    fn list_nodes(&self) -> Vec<Hash> {
        let prefix = format!("{}blocks/", self.prefix);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let mut nodes = Vec::new();

        self.runtime.block_on(async {
            let mut continuation_token = None;
            loop {
                let mut request = client
                    .list_objects_v2()
                    .bucket(&bucket)
                    .prefix(&prefix);

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                match request.send().await {
                    Ok(response) => {
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
                    Err(_) => break,
                }
            }
        });

        nodes
    }

    fn count_nodes(&self) -> usize {
        self.list_nodes().len()
    }
}

impl Remote for S3BlockStore {
    fn url(&self) -> String {
        format!("s3://{}/{}", self.bucket, self.prefix)
    }

    fn list_refs(&self) -> Vec<String> {
        let prefix = format!("{}refs/", self.prefix);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let mut refs = Vec::new();

        self.runtime.block_on(async {
            let mut continuation_token = None;
            loop {
                let mut request = client
                    .list_objects_v2()
                    .bucket(&bucket)
                    .prefix(&prefix);

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                match request.send().await {
                    Ok(response) => {
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
                    Err(_) => break,
                }
            }
        });

        refs
    }

    fn get_ref_commit(&self, ref_name: &str) -> Option<String> {
        let key = self.ref_key(ref_name);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        self.runtime.block_on(async {
            match client.get_object().bucket(&bucket).key(&key).send().await {
                Ok(response) => {
                    response.body.collect().await.ok().and_then(|b| {
                        String::from_utf8(b.to_vec()).ok().map(|s| s.trim().to_string())
                    })
                }
                Err(_) => None,
            }
        })
    }

    fn update_ref(&self, ref_name: &str, old_hash: Option<&str>, new_hash: &str) -> bool {
        let key = self.ref_key(ref_name);
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let new_hash = new_hash.to_string();

        self.runtime.block_on(async {
            // First, check current value if old_hash is specified
            if let Some(expected_old) = old_hash {
                match client.get_object().bucket(&bucket).key(&key).send().await {
                    Ok(response) => {
                        let current = response.body.collect().await.ok().and_then(|b| {
                            String::from_utf8(b.to_vec()).ok().map(|s| s.trim().to_string())
                        });

                        if current.as_deref() != Some(expected_old) {
                            return false;
                        }
                    }
                    Err(e) => {
                        // If error is NoSuchKey, that's a conflict (expected old but not present)
                        let is_not_found = e.as_service_error()
                            .map(|se| se.is_no_such_key())
                            .unwrap_or(false);
                        if is_not_found {
                            return false;
                        }
                        // Other errors - return false to be safe
                        return false;
                    }
                }
            } else {
                // old_hash is None - ref should not exist
                match client.head_object().bucket(&bucket).key(&key).send().await {
                    Ok(_) => return false, // Ref already exists
                    Err(_) => {} // Good, ref doesn't exist
                }
            }

            // Update the ref
            client
                .put_object()
                .bucket(&bucket)
                .key(&key)
                .body(new_hash.into_bytes().into())
                .send()
                .await
                .is_ok()
        })
    }

    fn put_commit(&self, commit_hash: &Hash, commit: Commit) {
        let key = self.commit_key(commit_hash);
        let data = serde_json::to_vec(&commit).expect("Failed to serialize commit");
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
                .ok();
        });

        // Update cache
        let mut cache = self.commit_cache.lock().unwrap();
        cache.insert(commit_hash.clone(), commit);
    }

    fn get_commit(&self, commit_hash: &Hash) -> Option<Commit> {
        // Check cache first
        {
            let cache = self.commit_cache.lock().unwrap();
            if let Some(commit) = cache.get(commit_hash) {
                return Some(commit.clone());
            }
        }

        let key = self.commit_key(commit_hash);
        let client = self.client.clone();
        let bucket = self.bucket.clone();

        let result = self.runtime.block_on(async {
            match client.get_object().bucket(&bucket).key(&key).send().await {
                Ok(response) => {
                    response.body.collect().await.ok().and_then(|b| {
                        serde_json::from_slice::<Commit>(&b.to_vec()).ok()
                    })
                }
                Err(_) => None,
            }
        });

        if let Some(ref commit) = result {
            // Update cache
            let mut cache = self.commit_cache.lock().unwrap();
            cache.insert(commit_hash.clone(), commit.clone());
        }

        result
    }

    fn get_parents(&self, commit_hash: &Hash) -> Vec<Hash> {
        self.get_commit(commit_hash)
            .map(|c| c.parents)
            .unwrap_or_default()
    }

    fn set_ref(&self, name: &str, commit_hash: &Hash) {
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
                .ok();
        });
    }

    fn get_ref(&self, name: &str) -> Option<Hash> {
        self.get_ref_commit(name)
            .and_then(|hex| hex::decode(&hex).ok())
    }
}
