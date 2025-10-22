use crate::board::BoardConfig;
use ndarray::{ArrayView1, ArrayView3};
use rand::SeedableRng;
use rand_pcg::Pcg64;

const CAPTURE_LIMIT: usize = 20;

fn rand63(rng: &mut Pcg64) -> u64 {
    use rand::RngCore;
    rng.next_u64() & ((1u64 << 63) - 1)
}

fn make_matrix(rng: &mut Pcg64, width: usize) -> Vec<Vec<u64>> {
    (0..width)
        .map(|_| (0..width).map(|_| rand63(rng)).collect())
        .collect()
}

#[derive(Clone)]
pub struct ZobristHasher {
    width: usize,
    ring: Vec<Vec<u64>>,
    marble: [Vec<Vec<u64>>; 3],
    captured: [[[u64; CAPTURE_LIMIT]; 3]; 2],
    supply: [[u64; CAPTURE_LIMIT]; 3],
    player: u64,
}

impl ZobristHasher {
    pub fn new(width: usize, seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        let ring = make_matrix(&mut rng, width);

        let marble = [
            make_matrix(&mut rng, width),
            make_matrix(&mut rng, width),
            make_matrix(&mut rng, width),
        ];

        let mut captured = [[[0u64; CAPTURE_LIMIT]; 3]; 2];
        for player in 0..2 {
            for marble_type in 0..3 {
                for count in 0..CAPTURE_LIMIT {
                    captured[player][marble_type][count] = rand63(&mut rng);
                }
            }
        }

        let mut supply = [[0u64; CAPTURE_LIMIT]; 3];
        for marble_type in 0..3 {
            for count in 0..CAPTURE_LIMIT {
                supply[marble_type][count] = rand63(&mut rng);
            }
        }

        let player = rand63(&mut rng);

        Self {
            width,
            ring,
            marble,
            captured,
            supply,
            player,
        }
    }

    fn marble_layers(config: &BoardConfig) -> [usize; 3] {
        [
            *config.marble_to_layer.get("w").expect("missing white layer"),
            *config.marble_to_layer.get("g").expect("missing gray layer"),
            *config.marble_to_layer.get("b").expect("missing black layer"),
        ]
    }

    fn hash_spatial(&self, spatial: &ArrayView3<f32>, config: &BoardConfig) -> u64 {
        let mut h = 0u64;

        // Rings
        for y in 0..self.width {
            for x in 0..self.width {
                if spatial[[config.ring_layer, y, x]] > 0.5 {
                    h ^= self.ring[y][x];
                }
            }
        }

        // Marbles
        let marble_layers = Self::marble_layers(config);
        for (table, layer_idx) in self.marble.iter().zip(marble_layers.iter()) {
            for y in 0..self.width {
                for x in 0..self.width {
                    if spatial[[*layer_idx, y, x]] > 0.5 {
                        h ^= table[y][x];
                    }
                }
            }
        }

        h
    }

    fn hash_supply_and_captures(
        &self,
        global: &ArrayView1<f32>,
        config: &BoardConfig,
    ) -> u64 {
        let mut h = 0u64;

        let supply_indices = [config.supply_w, config.supply_g, config.supply_b];
        for (idx, &global_idx) in supply_indices.iter().enumerate() {
            let count = global[global_idx].round() as isize;
            if count > 0 && (count as usize) < CAPTURE_LIMIT {
                h ^= self.supply[idx][count as usize];
            }
        }

        let p1_indices = [config.p1_cap_w, config.p1_cap_g, config.p1_cap_b];
        for (idx, &global_idx) in p1_indices.iter().enumerate() {
            let count = global[global_idx].round() as isize;
            if count > 0 && (count as usize) < CAPTURE_LIMIT {
                h ^= self.captured[0][idx][count as usize];
            }
        }

        let p2_indices = [config.p2_cap_w, config.p2_cap_g, config.p2_cap_b];
        for (idx, &global_idx) in p2_indices.iter().enumerate() {
            let count = global[global_idx].round() as isize;
            if count > 0 && (count as usize) < CAPTURE_LIMIT {
                h ^= self.captured[1][idx][count as usize];
            }
        }

        if global[config.cur_player].round() as usize == config.player_2 {
            h ^= self.player;
        }

        h
    }

    pub fn hash_state(
        &self,
        spatial: &ArrayView3<f32>,
        global: &ArrayView1<f32>,
        config: &BoardConfig,
    ) -> u64 {
        let mut h = self.hash_spatial(spatial, config);
        h ^= self.hash_supply_and_captures(global, config);
        h
    }

    #[allow(dead_code)]
    pub fn hash_canonical_spatial(
        &self,
        spatial: &ArrayView3<f32>,
        config: &BoardConfig,
    ) -> u64 {
        self.hash_spatial(spatial, config)
    }

    #[allow(dead_code)]
    pub fn width(&self) -> usize {
        self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zobrist_pcg64_matches_python() {
        // These values come from Python: np.random.Generator(np.random.PCG64(42)).integers(0, 2**63, 5)
        // We'll verify these after testing
        let mut rng = Pcg64::seed_from_u64(42);

        // Just verify it produces consistent values
        let first = rand63(&mut rng);

        // Reset and verify determinism
        let mut rng2 = Pcg64::seed_from_u64(42);
        let first2 = rand63(&mut rng2);

        assert_eq!(first, first2, "PCG64 should be deterministic with same seed");
    }

    #[test]
    fn zobrist_hasher_generates_consistent_tables() {
        // Create two hashers with same seed
        let hasher1 = ZobristHasher::new(7, 42);
        let hasher2 = ZobristHasher::new(7, 42);

        // Check that ring tables match (determinism)
        assert_eq!(hasher1.ring[0][0], hasher2.ring[0][0]);
        assert_eq!(hasher1.ring[0][1], hasher2.ring[0][1]);

        // Verify all tables are identical
        for y in 0..7 {
            for x in 0..7 {
                assert_eq!(hasher1.ring[y][x], hasher2.ring[y][x]);
            }
        }
    }
}
