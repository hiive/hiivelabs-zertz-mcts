//! # Algebraic Notation Module
//!
//! Converts between array coordinates (y, x) and algebraic notation (e.g., "A1", "D4").
//!
//! ## Hexagonal Board Layout
//!
//! The Zertz board is hexagonal, not rectangular. The array stores the board with:
//! - y=0 at the top, y increases downward
//! - x=0 at the left, x increases rightward
//! - The middle row (y ≈ width/2) is the longest, containing all columns
//! - Rows above and below the middle are shorter
//!
//! ## Coordinate System
//!
//! - **Columns**: A, B, C, ... (left to right, x-axis)
//! - **Rows**: Row numbers increase as you move up-right in the hexagon
//!
//! The row number depends on BOTH x and y coordinates:
//! - `row = min(width, width/2 + x + 1) - y`
//!
//! ## Board Sizes
//!
//! - **37 rings**: 7x7 array, columns A-G
//!
//!       A4 B5 C6 D7 .. .. ..
//!       A3 B4 C5 D6 E6 .. ..
//!       A2 B3 C4 D5 E5 F5 ..
//!       A1 B2 C3 D4 E4 F4 G4
//!       .. B1 C2 D3 E3 F3 G3
//!       .. .. C1 D2 E2 F2 G2
//!       .. .. .. D1 E1 F1 G1
//!
//! - **49 rings**: 8x8 array, columns A-H
//!
//!       A5 B6 C7 D8 .. .. .. ..
//!       A4 B5 C6 D7 E7 .. .. ..
//!       A3 B4 C5 D6 E6 F6 .. ..
//!       A2 B3 C4 D5 E5 F5 G5 ..
//!       A1 B2 C3 D4 E4 F4 G4 H4
//!       .. B1 C2 D3 E3 F3 G3 H3
//!       .. .. C1 D2 E2 F2 G2 H2
//!       .. .. .. D1 E1 F1 G1 H1
//!
//! - **61 rings**: 9x9 array, columns A-J (note: 'I' is skipped)
//!
//!       A5 B6 C7 D8 E9 .. .. .. ..
//!       A4 B5 C6 D7 E8 F8 .. .. ..
//!       A3 B4 C5 D6 E7 F7 G7 .. ..
//!       A2 B3 C4 D5 E6 F6 G6 H6 ..
//!       A1 B2 C3 D4 E5 F5 G5 H5 J5
//!       .. B1 C2 D3 E4 F4 G4 H4 J4
//!       .. .. C1 D2 E3 F3 G3 H3 J3
//!       .. .. .. D1 E2 F2 G2 H2 J2
//!       .. .. .. .. E1 F1 G1 H1 J1
//!
//! ## Examples
//!
//! For a 37-ring (7x7) board:
//! - A1 = (3, 0) - left side of middle row
//! - D4 = (3, 3) - center of board
//! - G4 = (3, 6) - right side of middle row
//! - G1 = (6, 6) - bottom-right corner
//! - A4 = (0, 0) - top-left corner

use super::board::BoardConfig;

/// Get the column letters for a given board width
fn get_column_letters(width: usize) -> Result<Vec<char>, String> {
    match width {
        7 => Ok("ABCDEFG".chars().collect()),
        8 => Ok("ABCDEFGH".chars().collect()),
        9 => Ok("ABCDEFGHJ".chars().collect()), // Note: 'I' is skipped
        _ => Err(format!("Unsupported board width: {}", width)),
    }
}

/// Convert array coordinates (y, x) to algebraic notation (e.g., "A1")
///
/// # Arguments
///
/// * `y` - Row index (0 = top, increases downward)
/// * `x` - Column index (0 = leftmost, increases rightward)
/// * `width` - Board width (7, 8, or 9)
///
/// # Returns
///
/// `Ok(String)` with algebraic notation (e.g., "D4")
/// `Err(String)` if coordinates are out of bounds
///
/// # Examples
///
/// ```
/// use hiivelabs_mcts::games::zertz::notation::coordinate_to_algebraic;
///
/// // For a 7x7 board:
/// assert_eq!(coordinate_to_algebraic(3, 0, 7).unwrap(), "A1"); // left middle
/// assert_eq!(coordinate_to_algebraic(3, 3, 7).unwrap(), "D4"); // center
/// assert_eq!(coordinate_to_algebraic(0, 0, 7).unwrap(), "A4"); // top-left
/// ```
pub fn coordinate_to_algebraic(y: usize, x: usize, width: usize) -> Result<String, String> {
    if y >= width {
        return Err(format!("Row index {} out of bounds for width {}", y, width));
    }
    if x >= width {
        return Err(format!(
            "Column index {} out of bounds for width {}",
            x, width
        ));
    }

    // Validate that position is within hexagonal bounds
    let mid_y = width / 2;
    if y <= mid_y {
        // Upper half: x can be from 0 to (mid_y + y)
        if x > mid_y + y {
            return Err(format!(
                "Position ({}, {}) is outside hexagonal board (upper half)",
                y, x
            ));
        }
    } else {
        // Lower half: x must be at least (y - mid_y)
        if x < y - mid_y {
            return Err(format!(
                "Position ({}, {}) is outside hexagonal board (lower half)",
                y, x
            ));
        }
    }

    let columns = get_column_letters(width)?;
    let column = columns[x];

    // Hexagonal board formula: row depends on both x and y
    // The middle row (y = width/2) is the longest, containing all columns
    // Row number increases as you move up and to the right from the center
    let mid_y = width / 2;
    let max_row_for_x = width.min(mid_y + x + 1);
    let row = max_row_for_x - y;

    Ok(format!("{}{}", column, row))
}

/// Convert array coordinates to algebraic notation using BoardConfig
///
/// # Examples
///
/// ```
/// use hiivelabs_mcts::games::zertz::board::BoardConfig;
/// use hiivelabs_mcts::games::zertz::notation::coordinate_to_algebraic_with_config;
///
/// let config = BoardConfig::standard(37, 1).unwrap();
/// assert_eq!(coordinate_to_algebraic_with_config(3, 0, &config).unwrap(), "A1");
/// assert_eq!(coordinate_to_algebraic_with_config(3, 3, &config).unwrap(), "D4");
/// ```
pub fn coordinate_to_algebraic_with_config(
    y: usize,
    x: usize,
    config: &BoardConfig,
) -> Result<String, String> {
    coordinate_to_algebraic(y, x, config.width)
}

/// Parse algebraic notation (e.g., "A1") to array coordinates (y, x)
///
/// # Arguments
///
/// * `notation` - Algebraic notation string (e.g., "D4", "A1")
/// * `width` - Board width (7, 8, or 9)
///
/// # Returns
///
/// `Ok((y, x))` with array coordinates
/// `Err(String)` if notation is invalid or out of bounds
///
/// # Examples
///
/// ```
/// use hiivelabs_mcts::games::zertz::notation::algebraic_to_coordinate;
///
/// // For a 7x7 board:
/// assert_eq!(algebraic_to_coordinate("A1", 7).unwrap(), (3, 0)); // left middle
/// assert_eq!(algebraic_to_coordinate("D4", 7).unwrap(), (3, 3)); // center
/// assert_eq!(algebraic_to_coordinate("A4", 7).unwrap(), (0, 0)); // top-left
/// ```
pub fn algebraic_to_coordinate(notation: &str, width: usize) -> Result<(usize, usize), String> {
    let notation = notation.trim().to_uppercase();

    if notation.is_empty() {
        return Err("Empty notation string".to_string());
    }

    let columns = get_column_letters(width)?;

    // Parse column (first character)
    let column_char = notation
        .chars()
        .next()
        .ok_or_else(|| "Empty notation string".to_string())?;

    let x = columns
        .iter()
        .position(|&c| c == column_char)
        .ok_or_else(|| format!("Invalid column letter: {}", column_char))?;

    // Parse row (remaining characters)
    let row_str = &notation[1..];
    let row: usize = row_str
        .parse()
        .map_err(|_| format!("Invalid row number: {}", row_str))?;

    if row == 0 {
        return Err("Row number cannot be 0".to_string());
    }

    // Convert to array coordinates using inverse hexagonal formula
    // From: row = min(width, mid_y + x + 1) - y
    // We get: y = min(width, mid_y + x + 1) - row
    let mid_y = width / 2;
    let max_row_for_x = width.min(mid_y + x + 1);

    // Check if row number is valid for this column
    if row > max_row_for_x {
        return Err(format!(
            "Row {} is out of bounds for column {} (max row is {})",
            row, column_char, max_row_for_x
        ));
    }

    // Calculate y
    let y = max_row_for_x - row;

    if y >= width {
        return Err(format!(
            "Computed position ({}, {}) is out of bounds for {}x{} board",
            y, x, width, width
        ));
    }

    Ok((y, x))
}

/// Parse algebraic notation using BoardConfig
///
/// # Examples
///
/// ```
/// use hiivelabs_mcts::games::zertz::board::BoardConfig;
/// use hiivelabs_mcts::games::zertz::notation::algebraic_to_coordinate_with_config;
///
/// let config = BoardConfig::standard(37, 1).unwrap();
/// assert_eq!(algebraic_to_coordinate_with_config("A1", &config).unwrap(), (3, 0));
/// assert_eq!(algebraic_to_coordinate_with_config("D4", &config).unwrap(), (3, 3));
/// ```
pub fn algebraic_to_coordinate_with_config(
    notation: &str,
    config: &BoardConfig,
) -> Result<(usize, usize), String> {
    algebraic_to_coordinate(notation, config.width)
}
