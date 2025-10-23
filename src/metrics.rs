use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Performance metrics for MCTS search
///
/// This module is only compiled when the `metrics` feature is enabled.
/// It tracks various performance statistics with minimal overhead using atomics.
#[cfg(feature = "metrics")]
#[derive(Debug)]
pub struct MCTSMetrics {
    // Transposition table metrics
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,

    // Node expansion metrics
    pub nodes_expanded: AtomicU64,
    pub nodes_reused: AtomicU64, // From transposition table

    // Search depth metrics
    pub max_depth_reached: AtomicUsize,
    pub total_depth: AtomicU64,     // Sum of all simulation depths
    pub simulation_count: AtomicU64, // Number of simulations

    // Performance metrics
    pub total_iterations: AtomicU64,
    pub search_start: Option<Instant>,
    pub search_duration_ns: AtomicU64,

    // Per-phase timing (optional, in nanoseconds)
    pub selection_time_ns: AtomicU64,
    pub expansion_time_ns: AtomicU64,
    pub simulation_time_ns: AtomicU64,
    pub backprop_time_ns: AtomicU64,
}

#[cfg(feature = "metrics")]
impl MCTSMetrics {
    pub fn new() -> Self {
        Self {
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            nodes_expanded: AtomicU64::new(0),
            nodes_reused: AtomicU64::new(0),
            max_depth_reached: AtomicUsize::new(0),
            total_depth: AtomicU64::new(0),
            simulation_count: AtomicU64::new(0),
            total_iterations: AtomicU64::new(0),
            search_start: None,
            search_duration_ns: AtomicU64::new(0),
            selection_time_ns: AtomicU64::new(0),
            expansion_time_ns: AtomicU64::new(0),
            simulation_time_ns: AtomicU64::new(0),
            backprop_time_ns: AtomicU64::new(0),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn start_search(&mut self) {
        self.search_start = Some(Instant::now());
    }

    pub fn end_search(&self) {
        if let Some(start) = self.search_start {
            let duration = start.elapsed().as_nanos() as u64;
            self.search_duration_ns.store(duration, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_node_expanded(&self) {
        self.nodes_expanded.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_node_reused(&self) {
        self.nodes_reused.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_iteration(&self) {
        self.total_iterations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_simulation(&self, depth: usize) {
        self.simulation_count.fetch_add(1, Ordering::Relaxed);
        self.total_depth.fetch_add(depth as u64, Ordering::Relaxed);

        // Update max depth if needed
        let mut current_max = self.max_depth_reached.load(Ordering::Relaxed);
        while depth > current_max {
            match self.max_depth_reached.compare_exchange_weak(
                current_max,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    #[inline(always)]
    pub fn record_selection_time(&self, nanos: u64) {
        self.selection_time_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_expansion_time(&self, nanos: u64) {
        self.expansion_time_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_simulation_time(&self, nanos: u64) {
        self.simulation_time_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn record_backprop_time(&self, nanos: u64) {
        self.backprop_time_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    // Computed metrics

    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 {
            (hits / total) * 100.0
        } else {
            0.0
        }
    }

    pub fn avg_depth(&self) -> f64 {
        let total = self.total_depth.load(Ordering::Relaxed) as f64;
        let count = self.simulation_count.load(Ordering::Relaxed) as f64;
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }

    pub fn iterations_per_second(&self) -> f64 {
        let iterations = self.total_iterations.load(Ordering::Relaxed) as f64;
        let duration_ns = self.search_duration_ns.load(Ordering::Relaxed) as f64;
        if duration_ns > 0.0 {
            iterations / (duration_ns / 1_000_000_000.0)
        } else {
            0.0
        }
    }

    pub fn reuse_rate(&self) -> f64 {
        let reused = self.nodes_reused.load(Ordering::Relaxed) as f64;
        let expanded = self.nodes_expanded.load(Ordering::Relaxed) as f64;
        let total = reused + expanded;
        if total > 0.0 {
            (reused / total) * 100.0
        } else {
            0.0
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "transposition_table": {{
    "cache_hits": {},
    "cache_misses": {},
    "cache_hit_rate_percent": {:.2}
  }},
  "nodes": {{
    "expanded": {},
    "reused": {},
    "reuse_rate_percent": {:.2}
  }},
  "search_depth": {{
    "max_depth": {},
    "avg_depth": {:.2}
  }},
  "performance": {{
    "total_iterations": {},
    "iterations_per_second": {:.0},
    "search_duration_ms": {:.2}
  }},
  "timing_breakdown_ms": {{
    "selection": {:.2},
    "expansion": {:.2},
    "simulation": {:.2},
    "backpropagation": {:.2}
  }}
}}"#,
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
            self.cache_hit_rate(),
            self.nodes_expanded.load(Ordering::Relaxed),
            self.nodes_reused.load(Ordering::Relaxed),
            self.reuse_rate(),
            self.max_depth_reached.load(Ordering::Relaxed),
            self.avg_depth(),
            self.total_iterations.load(Ordering::Relaxed),
            self.iterations_per_second(),
            self.search_duration_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.selection_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.expansion_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.simulation_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.backprop_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        )
    }

    pub fn print_summary(&self) {
        eprintln!("\n=== MCTS Performance Metrics ===");
        eprintln!("Cache hit rate: {:.2}%", self.cache_hit_rate());
        eprintln!(
            "Nodes expanded: {}",
            self.nodes_expanded.load(Ordering::Relaxed)
        );
        eprintln!(
            "Nodes reused: {} ({:.2}%)",
            self.nodes_reused.load(Ordering::Relaxed),
            self.reuse_rate()
        );
        eprintln!(
            "Max depth reached: {}",
            self.max_depth_reached.load(Ordering::Relaxed)
        );
        eprintln!("Avg depth: {:.2}", self.avg_depth());
        eprintln!(
            "Total iterations: {}",
            self.total_iterations.load(Ordering::Relaxed)
        );
        eprintln!("Iterations/sec: {:.0}", self.iterations_per_second());
        eprintln!(
            "Search duration: {:.2}ms",
            self.search_duration_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0
        );
        eprintln!("=================================\n");
    }
}

#[cfg(feature = "metrics")]
impl Default for MCTSMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Zero-size type when metrics are disabled
#[cfg(not(feature = "metrics"))]
#[derive(Debug, Clone, Copy)]
pub struct MCTSMetrics;

#[cfg(not(feature = "metrics"))]
impl MCTSMetrics {
    #[inline(always)]
    pub fn new() -> Self {
        MCTSMetrics
    }

    #[inline(always)]
    pub fn reset(&mut self) {}

    #[inline(always)]
    pub fn start_search(&mut self) {}

    #[inline(always)]
    pub fn end_search(&self) {}

    #[inline(always)]
    pub fn record_cache_hit(&self) {}

    #[inline(always)]
    pub fn record_cache_miss(&self) {}

    #[inline(always)]
    pub fn record_node_expanded(&self) {}

    #[inline(always)]
    pub fn record_node_reused(&self) {}

    #[inline(always)]
    pub fn record_iteration(&self) {}

    #[inline(always)]
    pub fn record_simulation(&self, _depth: usize) {}

    #[inline(always)]
    pub fn record_selection_time(&self, _nanos: u64) {}

    #[inline(always)]
    pub fn record_expansion_time(&self, _nanos: u64) {}

    #[inline(always)]
    pub fn record_simulation_time(&self, _nanos: u64) {}

    #[inline(always)]
    pub fn record_backprop_time(&self, _nanos: u64) {}
}

#[cfg(not(feature = "metrics"))]
impl Default for MCTSMetrics {
    fn default() -> Self {
        Self::new()
    }
}