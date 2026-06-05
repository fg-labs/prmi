// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

//! Reader for bwa's `bntseq` forward `.pac` (2-bit, 4 bases/byte, MSB-first).
//! bwa substitutes ambiguous bases (N) with a random base at pack time, so the
//! bases read here already carry bwa's N substitution — building the 2× SA from
//! this source is what makes prmi byte-identical to bwa's FM-index on real refs.

use std::path::Path;

use crate::error::{Error, Result};

/// Read bwa's forward `.pac` at `path`, returning `(bases, l_pac)` where `bases`
/// is one byte per base (values 0..=3, A=0/C=1/G=2/T=3) of length `l_pac`.
pub fn read_bwa_pac_forward(path: &Path) -> Result<(Vec<u8>, u64)> {
    let raw = std::fs::read(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    if raw.len() < 2 {
        return Err(Error::InvalidInput {
            detail: format!(
                "pac file {} too short ({} bytes)",
                path.display(),
                raw.len()
            ),
        });
    }
    let r = *raw.last().unwrap() as u64; // l_pac % 4
    if r > 3 {
        return Err(Error::InvalidInput {
            detail: format!(
                "corrupt .pac {}: trailing byte {r} is not a valid l_pac%4 (0..=3)",
                path.display()
            ),
        });
    }
    let data_bytes = raw.len() - 1;
    let l_pac = (data_bytes as u64 - 1) * 4 + r;
    // l_pac must fit in the data region: ceil(l_pac/4) <= data_bytes.
    if (l_pac as usize).div_ceil(4) > data_bytes {
        return Err(Error::InvalidInput {
            detail: format!(
                "corrupt .pac {}: recovered l_pac={l_pac} exceeds data region ({data_bytes} bytes)",
                path.display()
            ),
        });
    }
    let data = &raw[..data_bytes];
    let mut bases = Vec::with_capacity(l_pac as usize);
    for i in 0..l_pac as usize {
        let byte = data[i >> 2];
        let shift = (3 - (i & 3)) * 2;
        bases.push((byte >> shift) & 3);
    }
    Ok((bases, l_pac))
}

/// SHA-256 (hex) of the `.pac` file's raw bytes — provenance to assert the prmi
/// sidecar and the FMI were built from the identical `.pac`.
pub fn pac_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let raw = std::fs::read(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut h = Sha256::new();
    h.update(&raw);
    Ok(format!("{:x}", h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror bwa's writer: pack `bases` (0..=3) 4-per-byte MSB-first, pad the
    /// data region to `l_pac/4 + 1` bytes, append the `l_pac % 4` trailing byte.
    fn write_bwa_pac(path: &Path, bases: &[u8]) {
        let l_pac = bases.len();
        let data_bytes = l_pac / 4 + 1;
        let mut buf = vec![0u8; data_bytes];
        for (i, &b) in bases.iter().enumerate() {
            let shift = (3 - (i & 3)) * 2;
            buf[i >> 2] |= b << shift;
        }
        buf.push((l_pac % 4) as u8);
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn roundtrip_various_lengths() {
        let dir = tempfile::tempdir().unwrap();
        for &n in &[1usize, 3, 4, 5, 8, 17, 33] {
            let orig: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
            let p = dir.path().join(format!("r{n}.pac"));
            write_bwa_pac(&p, &orig);
            let (bases, l_pac) = read_bwa_pac_forward(&p).unwrap();
            assert_eq!(l_pac as usize, n, "l_pac recovery for n={n}");
            assert_eq!(bases, orig, "bases for n={n}");
        }
    }

    #[test]
    fn rejects_corrupt_trailing_byte() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.pac");
        // 2 data bytes + a bogus tail byte 0xFF (not 0..=3).
        std::fs::write(&p, [0xABu8, 0xCD, 0xFF]).unwrap();
        assert!(read_bwa_pac_forward(&p).is_err());
    }

    #[test]
    fn pac_sha256_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.pac");
        write_bwa_pac(&p, &[0, 1, 2, 3, 0]);
        let a = pac_sha256(&p).unwrap();
        let b = pac_sha256(&p).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
