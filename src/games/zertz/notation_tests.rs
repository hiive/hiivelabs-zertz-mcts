#[cfg(test)]
mod tests {
    use super::super::notation::*;
    use super::super::board::BoardConfig;

    #[test]
    fn test_coordinate_to_algebraic_37() {
        // Corner cases
        assert_eq!(coordinate_to_algebraic(3, 0, 7).unwrap(), "A1"); // bottom-left
        assert_eq!(coordinate_to_algebraic(3, 6, 7).unwrap(), "G4"); // top-right
        assert_eq!(coordinate_to_algebraic(0, 0, 7).unwrap(), "A4"); // top-left
        assert_eq!(coordinate_to_algebraic(6, 6, 7).unwrap(), "G1"); // bottom-right

        // Center
        assert_eq!(coordinate_to_algebraic(3, 3, 7).unwrap(), "D4");

        // Other positions
        assert_eq!(coordinate_to_algebraic(2, 4, 7).unwrap(), "E5");
        assert_eq!(coordinate_to_algebraic(4, 1, 7).unwrap(), "B1");
    }

    #[test]
    fn test_coordinate_to_algebraic_49() {
        assert_eq!(coordinate_to_algebraic(4, 0, 8).unwrap(), "A1"); // middle row left
        assert_eq!(coordinate_to_algebraic(0, 3, 8).unwrap(), "D8"); // top row
        assert_eq!(coordinate_to_algebraic(4, 3, 8).unwrap(), "D4"); // center
    }

    #[test]
    fn test_coordinate_to_algebraic_61() {
        // Note: 'I' is skipped, so J is the 9th column
        assert_eq!(coordinate_to_algebraic(4, 0, 9).unwrap(), "A1"); // middle row left
        assert_eq!(coordinate_to_algebraic(0, 4, 9).unwrap(), "E9"); // top row
        assert_eq!(coordinate_to_algebraic(4, 4, 9).unwrap(), "E5"); // center
    }

    #[test]
    fn test_coordinate_to_algebraic_out_of_bounds() {
        assert!(coordinate_to_algebraic(7, 0, 7).is_err());
        assert!(coordinate_to_algebraic(0, 7, 7).is_err());
        assert!(coordinate_to_algebraic(100, 100, 7).is_err());
    }

    #[test]
    fn test_algebraic_to_coordinate_37() {
        assert_eq!(algebraic_to_coordinate("A1", 7).unwrap(), (3, 0));
        assert_eq!(algebraic_to_coordinate("D7", 7).unwrap(), (0, 3));
        assert_eq!(algebraic_to_coordinate("D4", 7).unwrap(), (3, 3));
        assert_eq!(algebraic_to_coordinate("E5", 7).unwrap(), (2, 4));
        assert_eq!(algebraic_to_coordinate("B2", 7).unwrap(), (3, 1));
    }

    #[test]
    fn test_algebraic_to_coordinate_case_insensitive() {
        assert_eq!(algebraic_to_coordinate("a1", 7).unwrap(), (3, 0));
        assert_eq!(algebraic_to_coordinate("d4", 7).unwrap(), (3, 3));
        assert_eq!(algebraic_to_coordinate("d7", 7).unwrap(), (0, 3));
    }

    #[test]
    fn test_algebraic_to_coordinate_48() {
        assert_eq!(algebraic_to_coordinate("A1", 8).unwrap(), (4, 0));
        assert_eq!(algebraic_to_coordinate("D8", 8).unwrap(), (0, 3));
        assert_eq!(algebraic_to_coordinate("D4", 8).unwrap(), (4, 3));
    }

    #[test]
    fn test_algebraic_to_coordinate_61() {
        assert_eq!(algebraic_to_coordinate("A1", 9).unwrap(), (4, 0));
        assert_eq!(algebraic_to_coordinate("E9", 9).unwrap(), (0, 4));
        assert_eq!(algebraic_to_coordinate("E5", 9).unwrap(), (4, 4));
    }

    #[test]
    fn test_algebraic_to_coordinate_invalid() {
        assert!(algebraic_to_coordinate("", 7).is_err());
        assert!(algebraic_to_coordinate("Z1", 7).is_err()); // Invalid column
        assert!(algebraic_to_coordinate("A0", 7).is_err()); // Row 0 invalid
        assert!(algebraic_to_coordinate("A8", 7).is_err()); // Out of bounds
        assert!(algebraic_to_coordinate("H1", 7).is_err()); // H doesn't exist on 7x7
        assert!(algebraic_to_coordinate("I1", 9).is_err()); // I is skipped on 9x9
        assert!(algebraic_to_coordinate("ABC", 7).is_err()); // Invalid format
        assert!(algebraic_to_coordinate("A", 7).is_err()); // Missing row
    }

    #[test]
    fn test_roundtrip_37() {
        // Only test valid hexagonal positions
        let mid_y = 7 / 2;
        for y in 0..7 {
            for x in 0..7 {
                // Check if position is valid in hexagonal board
                let valid = if y <= mid_y {
                    x <= mid_y + y
                } else {
                    x >= y - mid_y
                };

                if valid {
                    let algebraic = coordinate_to_algebraic(y, x, 7).unwrap();
                    let (y2, x2) = algebraic_to_coordinate(&algebraic, 7).unwrap();
                    assert_eq!((y, x), (y2, x2), "Roundtrip failed for ({}, {}) -> {}", y, x, algebraic);
                }
            }
        }
    }

    #[test]
    fn test_roundtrip_49() {
        // Only test valid hexagonal positions
        let mid_y = 8 / 2;
        for y in 0..8 {
            for x in 0..8 {
                // Check if position is valid in hexagonal board
                let valid = if y <= mid_y {
                    x <= mid_y + y
                } else {
                    x >= y - mid_y
                };

                if valid {
                    let algebraic = coordinate_to_algebraic(y, x, 8).unwrap();
                    let (y2, x2) = algebraic_to_coordinate(&algebraic, 8).unwrap();
                    assert_eq!((y, x), (y2, x2), "Roundtrip failed for ({}, {}) -> {}", y, x, algebraic);
                }
            }
        }
    }

    #[test]
    fn test_roundtrip_61() {
        // Only test valid hexagonal positions
        let mid_y = 9 / 2;
        for y in 0..9 {
            for x in 0..9 {
                // Check if position is valid in hexagonal board
                let valid = if y <= mid_y {
                    x <= mid_y + y
                } else {
                    x >= y - mid_y
                };

                if valid {
                    let algebraic = coordinate_to_algebraic(y, x, 9).unwrap();
                    let (y2, x2) = algebraic_to_coordinate(&algebraic, 9).unwrap();
                    assert_eq!((y, x), (y2, x2), "Roundtrip failed for ({}, {}) -> {}", y, x, algebraic);
                }
            }
        }
    }

    #[test]
    fn test_with_config() {
        let config = BoardConfig::standard(37, 1).unwrap();

        assert_eq!(
            coordinate_to_algebraic_with_config(3, 0, &config).unwrap(),
            "A1"
        );
        assert_eq!(
            coordinate_to_algebraic_with_config(3, 3, &config).unwrap(),
            "D4"
        );

        assert_eq!(
            algebraic_to_coordinate_with_config("A1", &config).unwrap(),
            (3, 0)
        );
        assert_eq!(
            algebraic_to_coordinate_with_config("D4", &config).unwrap(),
            (3, 3)
        );
    }
}
