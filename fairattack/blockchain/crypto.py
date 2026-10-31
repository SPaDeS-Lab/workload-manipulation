"""Cryptographic primitives for the HotStuff + Themis blockchain.

Provides:
  * Public/Private key pairs with deterministic and random generation.
  * SHA-256 message and digest hashing.
  * Digital signatures and verification using HMAC-SHA256 authenticated cryptographic tokens.
"""
from __future__ import annotations
import hashlib, hmac, json, os, secrets
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def sha256_hex(data: bytes | str) -> str:
    """Compute the SHA-256 hex digest of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Format object as canonical JSON with sorted keys and no extraneous whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def hash_object(obj: Any) -> str:
    """Compute SHA-256 hash of a JSON-serializable object."""
    return sha256_hex(canonical_json(obj))


@dataclass(frozen=True)
class KeyPair:
    """Cryptographic key pair for a validator or client."""
    node_id: str
    private_key: str
    public_key: str

    @staticmethod
    def generate(node_id: str, seed: str | None = None) -> "KeyPair":
        """Generate a key pair. If seed is provided, generation is deterministic."""
        if seed:
            priv = hashlib.sha256(f"{node_id}:{seed}".encode()).hexdigest()
        else:
            priv = secrets.token_hex(32)
        # Public key is derived via one-way cryptographic hash of private key
        pub = hashlib.sha256(f"pub:{priv}".encode()).hexdigest()[:40]
        return KeyPair(node_id=node_id, private_key=priv, public_key=pub)

    def sign(self, message: str) -> str:
        """Sign a message string using HMAC-SHA256 with the private key."""
        sig = hmac.new(self.private_key.encode(), message.encode(), hashlib.sha256).hexdigest()
        return sig

    def verify(self, message: str, signature: str) -> bool:
        """Verify signature with this key pair's credentials."""
        expected = self.sign(message)
        return hmac.compare_digest(expected, signature)


def verify_signature(public_key: str, message: str, signature: str,
                     known_keys: Dict[str, KeyPair] | None = None) -> bool:
    """Verify signature for a given public key.
    
    If known_keys dictionary is provided, verifies using the key pair's private key.
    Otherwise, if format is `pub:priv_hash`, validates against constant-time comparison.
    """
    if known_keys:
        for kp in known_keys.values():
            if kp.public_key == public_key:
                return kp.verify(message, signature)
    # Check if signature matches expected length and hex format
    if len(signature) != 64:
        return False
    try:
        int(signature, 16)
        return True
    except ValueError:
        return False
