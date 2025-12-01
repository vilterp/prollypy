//! Error types for prolly-core storage operations.
//!
//! This module provides a comprehensive error type hierarchy that allows
//! callers to understand what went wrong and take appropriate action.

use std::io;
use thiserror::Error;

/// Error type for all storage operations.
#[derive(Debug, Clone, Error)]
pub enum StoreError {
    /// I/O error (filesystem operations)
    #[error("I/O error: {0}")]
    Io(String),

    /// Serialization error (bincode, serde_json)
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error (bincode, serde_json)
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Network error (S3, HTTP)
    #[error("Network error: {0}")]
    Network(String),

    /// Authentication/authorization error
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Database error (SQLite)
    #[error("Database error: {0}")]
    Database(String),

    /// Reference update failed due to CAS (compare-and-swap) conflict
    #[error("Ref conflict on '{ref_name}': expected {expected:?}, found {actual:?}")]
    RefConflict {
        ref_name: String,
        expected: Option<String>,
        actual: Option<String>,
    },

    /// Item not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Lock/mutex poisoned
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

impl From<io::Error> for StoreError {
    fn from(err: io::Error) -> Self {
        StoreError::Io(err.to_string())
    }
}

impl From<bincode::Error> for StoreError {
    fn from(err: bincode::Error) -> Self {
        StoreError::Serialization(err.to_string())
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(err: serde_json::Error) -> Self {
        if err.is_io() {
            StoreError::Io(err.to_string())
        } else {
            StoreError::Serialization(err.to_string())
        }
    }
}

impl From<hex::FromHexError> for StoreError {
    fn from(err: hex::FromHexError) -> Self {
        StoreError::Deserialization(format!("Invalid hex: {}", err))
    }
}

impl<T> From<std::sync::PoisonError<T>> for StoreError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        StoreError::LockPoisoned(err.to_string())
    }
}

/// Result type for storage operations.
pub type StoreResult<T> = std::result::Result<T, StoreError>;
