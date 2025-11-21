package prollycore

import (
	"crypto/sha256"
	"encoding/hex"
)

// Hash is a 16-byte content address
type Hash [16]byte

// EmptyHash is the zero hash
var EmptyHash Hash

// HashFromBytes creates a Hash from a byte slice
func HashFromBytes(b []byte) Hash {
	var h Hash
	copy(h[:], b)
	return h
}

// HashFromHex creates a Hash from a hex string
func HashFromHex(s string) (Hash, error) {
	b, err := hex.DecodeString(s)
	if err != nil {
		return Hash{}, err
	}
	return HashFromBytes(b), nil
}

// Hex returns the hex encoding of the hash
func (h Hash) Hex() string {
	return hex.EncodeToString(h[:])
}

// String returns the hex encoding
func (h Hash) String() string {
	return h.Hex()
}

// IsEmpty returns true if this is the zero hash
func (h Hash) IsEmpty() bool {
	return h == EmptyHash
}

// ComputeHash computes a SHA256 hash and returns first 16 bytes
func ComputeHash(data []byte) Hash {
	sum := sha256.Sum256(data)
	return HashFromBytes(sum[:16])
}
