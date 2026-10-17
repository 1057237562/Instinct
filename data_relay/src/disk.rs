//! Data-disk quota policy: pure arithmetic + decision logic (unit-tested),
//! with the actual SFTP cleanup driven by the runner.

/// Snapshot of the instance data disk.
#[derive(Debug, Clone)]
pub struct DiskUsage {
    pub total: u64,
    pub used: u64,
    pub source: &'static str,
}

impl DiskUsage {
    pub fn free(&self) -> u64 {
        self.total.saturating_sub(self.used)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub allowed: bool,
    pub projected_used: u64,
    pub limit: u64,
    pub message: String,
}

/// Quota check: `used + incoming` must stay below `total * max_ratio - reserved`.
/// Unknown totals (total == 0) allow the transfer with a warning — the
/// post-upload report still surfaces real usage.
pub fn plan(
    usage: &DiskUsage,
    incoming: u64,
    max_ratio: f64,
    reserved_gb: f64,
) -> Decision {
    let reserved = (reserved_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let limit = (usage.total as f64 * max_ratio) as u64;
    let limit = limit.saturating_sub(reserved);
    let projected = usage.used.saturating_add(incoming);
    if usage.total == 0 {
        return Decision {
            allowed: true,
            projected_used: projected,
            limit: 0,
            message: "磁盘总容量未知，跳过预检（上传后请查看用量报告）".into(),
        };
    }
    if projected <= limit {
        Decision {
            allowed: true,
            projected_used: projected,
            limit,
            message: format!(
                "预计已用 {} / 限额 {}",
                crate::model::human_bytes(projected),
                crate::model::human_bytes(limit)
            ),
        }
    } else {
        Decision {
            allowed: false,
            projected_used: projected,
            limit,
            message: format!(
                "超出数据盘限额：现有 {} + 待传 {} = {} > 限额 {}（清理托管目录或扩容数据盘）",
                crate::model::human_bytes(usage.used),
                crate::model::human_bytes(incoming),
                crate::model::human_bytes(projected),
                crate::model::human_bytes(limit)
            ),
        }
    }
}

/// After deleting `freed` bytes, does `used + incoming` fit?
pub fn fits_after_free(usage: &DiskUsage, freed: u64, incoming: u64, limit: u64) -> bool {
    usage.used.saturating_sub(freed).saturating_add(incoming) <= limit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(total_gb: u64, used_gb: u64) -> DiskUsage {
        DiskUsage {
            total: total_gb * 1024 * 1024 * 1024,
            used: used_gb * 1024 * 1024 * 1024,
            source: "test",
        }
    }

    #[test]
    fn allows_within_limit() {
        let d = plan(&usage(50, 30), 10 * GB, 0.92, 2.0);
        assert!(d.allowed);
        assert_eq!(d.projected_used, 40 * GB);
    }

    #[test]
    fn rejects_over_limit_with_message() {
        let d = plan(&usage(50, 44), 10 * GB, 0.92, 2.0);
        assert!(!d.allowed);
        assert!(d.message.contains("超出数据盘限额"));
    }

    #[test]
    fn zero_total_allows_with_warning() {
        let d = plan(&DiskUsage { total: 0, used: 0, source: "test" }, 100, 0.92, 2.0);
        assert!(d.allowed);
        assert!(d.message.contains("未知"));
    }

    #[test]
    fn freed_bytes_reopen_headroom() {
        let u = usage(50, 44);
        let incoming = 10 * GB;
        let d = plan(&u, incoming, 0.92, 2.0);
        assert!(!d.allowed);
        // limit = 46 - 2 = 44 GB; need (44 - freed) + 10 <= 44 → freed >= 10 GB
        assert!(!fits_after_free(&u, 8 * GB, incoming, d.limit));
        assert!(fits_after_free(&u, 10 * GB, incoming, d.limit));
    }

    const GB: u64 = 1024 * 1024 * 1024;
}
