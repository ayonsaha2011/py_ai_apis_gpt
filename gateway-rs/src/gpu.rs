use std::process::Command;

use regex::Regex;

use crate::models::GpuInfo;

pub fn list_gpus() -> anyhow::Result<Vec<GpuInfo>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Ok(Vec::new());
    }
    parse_nvidia_smi_csv(&String::from_utf8_lossy(&output.stdout))
}

pub fn parse_nvidia_smi_csv(text: &str) -> anyhow::Result<Vec<GpuInfo>> {
    let number = Regex::new(r"^\s*(\d+)\s*$")?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() != 6 {
            continue;
        }
        let parse_u64 = |s: &str| -> anyhow::Result<u64> {
            if let Some(caps) = number.captures(s) {
                Ok(caps[1].parse()?)
            } else {
                Ok(s.parse()?)
            }
        };
        out.push(GpuInfo {
            index: parts[0].parse()?,
            name: parts[1].to_owned(),
            total_mb: parse_u64(parts[2])?,
            used_mb: parse_u64(parts[3])?,
            free_mb: parse_u64(parts[4])?,
            utilization_pct: parts[5].parse()?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_csv() {
        let gpus = parse_nvidia_smi_csv("0, NVIDIA RTX, 32607, 2152, 30036, 11\n").unwrap();
        assert_eq!(gpus[0].free_mb, 30036);
    }
}
