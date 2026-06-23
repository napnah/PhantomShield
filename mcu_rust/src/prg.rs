//! AES-CTR 同步 PRG，字节级对齐 Python `mcu_core/prg_sync.py`。
//!
//! Python 实现要点（必须逐位复刻）：
//! ```text
//! nonce   = pack('>Q', counter) || 0x00*8        # 16 字节计数器块
//! block   = AES-128-ECB(key, nonce)              # CTR 首块 = ECB(计数器块)
//! rand    = block[0..8] 解释为大端 u64
//! next()  = rand % ring
//! ```
//! 即 `next()` 等价于「对 16 字节块 `BE64(counter) || zeros[8]` 做 AES-128 加密，
//! 取密文前 8 字节大端为 u64」。本实现复用 AES key schedule，可批量/并行生成，
//! 既与 Python 对齐又更高效。

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;

const TWO_POW_53: u64 = 1u64 << 53;

/// 同步伪随机数生成器（复用 AES key schedule）。
#[derive(Clone)]
pub struct PrgSync {
    cipher: Aes128,
    counter: u64,
}

impl PrgSync {
    /// 用 16 字节种子初始化。
    pub fn new(seed: &[u8; 16]) -> Self {
        let key = GenericArray::from_slice(seed);
        Self {
            cipher: Aes128::new(key),
            counter: 0,
        }
    }

    /// 生成原始 64 位随机字（对应 Python `next(2**64)`）。
    #[inline]
    pub fn raw64(&mut self) -> u64 {
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&self.counter.to_be_bytes());
        // block[8..16] 保持为 0，对应 Python 的 8 字节零填充
        let mut ga = GenericArray::clone_from_slice(&block);
        self.cipher.encrypt_block(&mut ga);
        self.counter = self.counter.wrapping_add(1);
        let mut out = [0u8; 8];
        out.copy_from_slice(&ga[..8]);
        u64::from_be_bytes(out)
    }

    /// 等价 Python `next()`（默认 ring = 2^64，整数环掩码）。
    #[inline]
    pub fn next(&mut self) -> u64 {
        self.raw64()
    }

    /// 等价 Python `next(ring)`，要求 `ring < 2^64`。
    #[inline]
    pub fn next_mod(&mut self, ring: u64) -> u64 {
        self.raw64() % ring
    }

    /// 等价 Python `next_unit()`：[0, 1) 的 53 位精度浮点。
    #[inline]
    pub fn next_unit(&mut self) -> f64 {
        let v = self.raw64() % TWO_POW_53;
        (v as f64) / (TWO_POW_53 as f64)
    }

    /// 等价 Python `next_real(high)`：[0, high) 浮点。
    #[inline]
    pub fn next_real(&mut self, high: f64) -> f64 {
        self.next_unit() * high
    }

    /// 当前计数器（用于按元素偏移定位序列）。
    #[inline]
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// 定位到指定计数器（不消耗）。
    #[inline]
    pub fn seek(&mut self, counter: u64) {
        self.counter = counter;
    }

    /// 无状态地取计数器 `counter` 对应的 64 位随机字。
    ///
    /// 因 `raw64(c)` 只依赖 `c`，该方法可在批处理中安全并行调用（`&self`）。
    #[inline]
    pub fn raw64_at(&self, counter: u64) -> u64 {
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&counter.to_be_bytes());
        let mut ga = GenericArray::clone_from_slice(&block);
        self.cipher.encrypt_block(&mut ga);
        let mut out = [0u8; 8];
        out.copy_from_slice(&ga[..8]);
        u64::from_be_bytes(out)
    }
}

/// 把 64 位随机字映射为 [0,1) 浮点（对齐 Python `next_unit`）。
#[inline]
pub fn unit_from(raw: u64) -> f64 {
    (raw % TWO_POW_53) as f64 / (TWO_POW_53 as f64)
}

/// 把 64 位随机字映射为 [0,high) 浮点（对齐 Python `next_real`）。
#[inline]
pub fn real_from(raw: u64, high: f64) -> f64 {
    unit_from(raw) * high
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_0_15() -> [u8; 16] {
        let mut s = [0u8; 16];
        for (i, b) in s.iter_mut().enumerate() {
            *b = i as u8;
        }
        s
    }

    #[test]
    fn golden_next_full() {
        // 黄金向量来自 Python: PRGSync(bytes(range(16))).next() x6
        let expected: [u64; 6] = [
            14312786200443706242,
            1376019470055311278,
            14370581582267350683,
            10433509245121005000,
            8777647769783999631,
            7958603945566699813,
        ];
        let mut p = PrgSync::new(&seed_0_15());
        for (i, e) in expected.iter().enumerate() {
            let got = p.next();
            assert_eq!(got, *e, "next() 第 {i} 个不匹配");
        }
    }

    #[test]
    fn golden_next_unit() {
        let expected: [f64; 4] = [
            0.038478626981359065,
            0.7688497987912706,
            0.45505498874259176,
            0.3522191572775961,
        ];
        let mut p = PrgSync::new(&seed_0_15());
        for (i, e) in expected.iter().enumerate() {
            let got = p.next_unit();
            assert!((got - e).abs() < 1e-15, "next_unit() 第 {i} 个不匹配: {got} vs {e}");
        }
    }

    #[test]
    fn golden_next_real_256() {
        let expected: [f64; 4] = [
            9.85052850722792,
            196.82554849056527,
            116.49407711810349,
            90.1681042630646,
        ];
        let mut p = PrgSync::new(&seed_0_15());
        for (i, e) in expected.iter().enumerate() {
            let got = p.next_real(256.0);
            assert!((got - e).abs() < 1e-10, "next_real(256) 第 {i} 个不匹配: {got} vs {e}");
        }
    }
}
