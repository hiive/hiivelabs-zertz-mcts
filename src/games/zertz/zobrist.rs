//! # Zobrist Hashing for Zertz
//!
//! Zobrist hashing implementation specific to the Zertz game.
//! Provides fast, incremental hash computation for Zertz game states.
//!
//! ## Algorithm
//!
//! **Zobrist hashing** assigns a random 64-bit number to each possible feature
//! (ring at position, marble at position, supply count, etc.). The hash of a state
//! is the XOR of all feature hashes that are "active" in that state.
//!
//! **Key properties**:
//! - **Fast**: XOR is O(1), hashing is O(board_size)
//! - **Incremental**: Can update hash with `hash_new = hash_old ^ feature_old ^ feature_new`
//! - **Low collision**: 64-bit hashes have ~1 in 2^64 collision rate
//! - **Deterministic**: Same seed produces same random tables
//!
//! ## Random Number Generation
//!
//! **PCG64 algorithm** (Permuted Congruential Generator):
//! - Fast, high-quality RNG with good statistical properties
//! - **Compatible with Python's `np.random.PCG64`** for cross-language testing
//! - Generates 63-bit values (not 64-bit) to avoid sign issues
//!
//! ## Hash Components
//!
//! Each game state is hashed by XORing random values for:
//! 1. **Ring positions**: Each (y, x) position where a ring exists
//! 2. **Marble positions**: Each (marble_type, y, x) where marble exists
//! 3. **Supply counts**: Number of each marble type in supply pool
//! 4. **Captured counts**: Number of each marble type captured by each player
//! 5. **Current player**: Single bit indicating whose turn it is
//!
//! ## Example
//!
//! ```text
//! State: ring at (0,0), white marble at (1,2), supply: 5W, player 1's turn
//!
//! hash = ring_table[0][0]
//!      ^ white_marble_table[1][2]
//!      ^ supply_white[5]
//!      ^ player_1_hash
//! ```

use crate::games::zertz::BoardConfig;
use ndarray::{ArrayView1, ArrayView3};
use rand::SeedableRng;
use rand_pcg::Pcg64;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Maximum number of marbles of one type that can be in supply or captured
///
/// This limits the size of our Zobrist tables. 20 is more than enough for Zèrtz
/// (standard game has 6W + 8G + 10B = 24 total marbles).
const CAPTURE_LIMIT: usize = 20;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate a random 63-bit number (not full 64-bit to avoid sign issues)
///
/// **Why 63 bits?**
/// - Avoids issues with signed/unsigned conversions
/// - Still provides ~9 × 10^18 possible values (extremely low collision rate)
/// - Matches common practice in Zobrist hashing implementations
pub(crate) fn rand63(rng: &mut Pcg64) -> u64 {
    use rand::RngCore;
    rng.next_u64() & ((1u64 << 63) - 1)
}

/// Generate a width × width matrix of random 63-bit values
///
/// Used to create Zobrist tables for ring and marble positions.
fn make_matrix(rng: &mut Pcg64, width: usize) -> Vec<Vec<u64>> {
    (0..width)
        .map(|_| (0..width).map(|_| rand63(rng)).collect())
        .collect()
}

// ============================================================================
// ZOBRIST HASHER
// ============================================================================

/// Zobrist hash table generator for game states
///
/// **Random tables structure**:
/// - `ring[y][x]`: Hash for ring at position (y, x)
/// - `marble[marble_type][y][x]`: Hash for marble at position
/// - `captured[player][marble_type][count]`: Hash for player having count marbles captured
/// - `supply[marble_type][count]`: Hash for count marbles in supply
/// - `player`: Hash bit for player 2's turn (player 1 = don't XOR this)
///
/// All tables are initialized with deterministic random values from PCG64.
#[derive(Clone)]
pub struct ZobristHasher {
    width: usize,
    pub(crate) ring: Vec<Vec<u64>>,            // [y][x]
    marble: [Vec<Vec<u64>>; 3],     // [marble_type][y][x]
    captured: [[[u64; CAPTURE_LIMIT]; 3]; 2], // [player][marble_type][count]
    supply: [[u64; CAPTURE_LIMIT]; 3], // [marble_type][count]
    player: u64,                    // Hash bit for player 2
}

impl ZobristHasher {
    /// Create a new Zobrist hasher with random tables
    ///
    /// **Initialization order** (important for determinism):
    /// 1. Ring position tables (width × width matrix)
    /// 2. Marble position tables (3 × width × width matrices)
    /// 3. Captured marble tables (2 players × 3 types × CAPTURE_LIMIT)
    /// 4. Supply marble tables (3 types × CAPTURE_LIMIT)
    /// 5. Player turn bit
    ///
    /// **Seed behavior**:
    /// - `Some(seed)`: Deterministic tables for testing/debugging
    /// - `None`: Non-deterministic tables using system RNG
    pub fn new(width: usize, seed: Option<u64>) -> Self {
        // Initialize RNG (PCG64 for Python compatibility)
        let mut rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_rng(&mut rand::rng()),
        };

        // Generate position tables
        let ring = make_matrix(&mut rng, width);

        let marble = [
            make_matrix(&mut rng, width), // White marbles
            make_matrix(&mut rng, width), // Gray marbles
            make_matrix(&mut rng, width), // Black marbles
        ];

        // Generate captured marble count tables
        // captured[player][marble_type][count] = hash for that count
        let mut captured = [[[0u64; CAPTURE_LIMIT]; 3]; 2];
        for player in 0..2 {
            for marble_type in 0..3 {
                for count in 0..CAPTURE_LIMIT {
                    captured[player][marble_type][count] = rand63(&mut rng);
                }
            }
        }

        // Generate supply count tables
        // supply[marble_type][count] = hash for that count in supply
        let mut supply = [[0u64; CAPTURE_LIMIT]; 3];
        for marble_type in 0..3 {
            for count in 0..CAPTURE_LIMIT {
                supply[marble_type][count] = rand63(&mut rng);
            }
        }

        // Player turn bit (XOR if player 2, don't XOR if player 1)
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

    /// Extract marble layer indices from config
    ///
    /// Returns `[white_layer, gray_layer, black_layer]` indices.
    fn marble_layers(config: &BoardConfig) -> [usize; 3] {
        [
            *config
                .marble_to_layer
                .get("w")
                .expect("missing white layer"),
            *config.marble_to_layer.get("g").expect("missing gray layer"),
            *config
                .marble_to_layer
                .get("b")
                .expect("missing black layer"),
        ]
    }

    /// Hash the spatial_state state (rings and marbles on board)
    ///
    /// **Algorithm**:
    /// 1. Start with hash = 0
    /// 2. For each ring present: XOR ring_table[y][x]
    /// 3. For each marble present: XOR marble_table[type][y][x]
    ///
    /// **Threshold 0.5**: Arrays use f32 (0.0 or 1.0), we check > 0.5 to handle
    /// floating point imprecision.
    fn hash_spatial_state(&self, spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> u64 {
        let mut h = 0u64;

        // Hash rings: XOR random value for each ring position
        for y in 0..self.width {
            for x in 0..self.width {
                if spatial_state[[config.ring_layer, y, x]] > 0.5 {
                    h ^= self.ring[y][x];
                }
            }
        }

        // Hash marbles: XOR random value for each marble position
        let marble_layers = Self::marble_layers(config);
        for (table, layer_idx) in self.marble.iter().zip(marble_layers.iter()) {
            for y in 0..self.width {
                for x in 0..self.width {
                    if spatial_state[[*layer_idx, y, x]] > 0.5 {
                        h ^= table[y][x];
                    }
                }
            }
        }

        h
    }

    /// Hash the global_state state (supply counts, captured counts, current player)
    ///
    /// **Components**:
    /// - **Supply counts**: XOR supply_table[marble_type][count]
    /// - **P1 captured**: XOR captured_table[0][marble_type][count]
    /// - **P2 captured**: XOR captured_table[1][marble_type][count]
    /// - **Current player**: XOR player bit if player 2's turn
    ///
    /// **Bounds checking**: Only hash counts < CAPTURE_LIMIT (defensive programming)
    fn hash_supply_and_captures(&self, global_state: &ArrayView1<f32>, config: &BoardConfig) -> u64 {
        let mut h = 0u64;

        // Hash supply counts for each marble type
        let supply_indices = [config.supply_w, config.supply_g, config.supply_b];
        for (idx, &global_state_idx) in supply_indices.iter().enumerate() {
            let count = global_state[global_state_idx].round() as isize;
            if count > 0 && (count as usize) < CAPTURE_LIMIT {
                h ^= self.supply[idx][count as usize];
            }
        }

        // Hash player 1's captured marbles
        let p1_indices = [config.p1_cap_w, config.p1_cap_g, config.p1_cap_b];
        for (idx, &global_state_idx) in p1_indices.iter().enumerate() {
            let count = global_state[global_state_idx].round() as isize;
            if count > 0 && (count as usize) < CAPTURE_LIMIT {
                h ^= self.captured[0][idx][count as usize];
            }
        }

        // Hash player 2's captured marbles
        let p2_indices = [config.p2_cap_w, config.p2_cap_g, config.p2_cap_b];
        for (idx, &global_state_idx) in p2_indices.iter().enumerate() {
            let count = global_state[global_state_idx].round() as isize;
            if count > 0 && (count as usize) < CAPTURE_LIMIT {
                h ^= self.captured[1][idx][count as usize];
            }
        }

        // Hash current player (XOR player bit only if player 2)
        if global_state[config.cur_player].round() as usize == config.player_2 {
            h ^= self.player;
        }

        h
    }

    /// Compute Zobrist hash for complete game state
    ///
    /// **Combines**:
    /// - spatial_state hash (rings + marbles on board)
    /// - global_state hash (supply + captured + player)
    ///
    /// **Usage**: Called by transposition table to hash canonical states.
    pub fn hash_state(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
        config: &BoardConfig,
    ) -> u64 {
        let mut h = self.hash_spatial_state(spatial_state, config);
        h ^= self.hash_supply_and_captures(global_state, config);
        h
    }

    #[allow(dead_code)]
    pub fn hash_canonical_spatial_state(&self, spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> u64 {
        self.hash_spatial_state(spatial_state, config)
    }

    #[allow(dead_code)]
    pub fn width(&self) -> usize {
        self.width
    }
}

