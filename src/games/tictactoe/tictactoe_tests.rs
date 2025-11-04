#[cfg(test)]
mod tests {
    use crate::game_trait::MCTSGame;
    use crate::games::tictactoe::{TicTacToeAction, TicTacToeGame};
    use ndarray::{Array1, Array3};

    #[test]
    fn test_initial_state() {
        let (spatial, global) = TicTacToeGame::initial_state();
        assert_eq!(spatial.shape(), &[2, 3, 3]);
        assert_eq!(global.len(), 1);
        assert_eq!(global[0], 0.0); // X starts
    }

    #[test]
    fn test_get_valid_actions_empty_board() {
        let game = TicTacToeGame::new();
        let (spatial, global) = TicTacToeGame::initial_state();
        let actions = game.get_valid_actions(&spatial.view(), &global.view());
        assert_eq!(actions.len(), 9); // All 9 cells are empty
    }

    #[test]
    fn test_apply_action() {
        let game = TicTacToeGame::new();
        let (mut spatial, mut global) = TicTacToeGame::initial_state();

        let action = TicTacToeAction { row: 1, col: 1 };
        let _ = game.apply_action(&mut spatial.view_mut(), &mut global.view_mut(), &action);

        // X should have marked center
        assert_eq!(spatial[[0, 1, 1]], 1.0);
        assert_eq!(spatial[[1, 1, 1]], 0.0);

        // Should switch to O
        assert_eq!(global[0], 1.0);

        // 8 cells remain
        let actions = game.get_valid_actions(&spatial.view(), &global.view());
        assert_eq!(actions.len(), 8);
    }

    #[test]
    fn test_x_wins_row() {
        let game = TicTacToeGame::new();
        let mut spatial = Array3::zeros((2, 3, 3));
        let global = Array1::from(vec![0.0]);

        // X wins top row
        spatial[[0, 0, 0]] = 1.0;
        spatial[[0, 0, 1]] = 1.0;
        spatial[[0, 0, 2]] = 1.0;

        assert!(game.is_terminal(&spatial.view(), &global.view()));
        assert_eq!(game.get_outcome(&spatial.view(), &global.view()), 1);
    }

    #[test]
    fn test_o_wins_diagonal() {
        let game = TicTacToeGame::new();
        let mut spatial = Array3::zeros((2, 3, 3));
        let global = Array1::from(vec![1.0]);

        // O wins main diagonal
        spatial[[1, 0, 0]] = 1.0;
        spatial[[1, 1, 1]] = 1.0;
        spatial[[1, 2, 2]] = 1.0;

        assert!(game.is_terminal(&spatial.view(), &global.view()));
        assert_eq!(game.get_outcome(&spatial.view(), &global.view()), -1);
    }

    #[test]
    fn test_draw() {
        let game = TicTacToeGame::new();
        let mut spatial = Array3::zeros((2, 3, 3));
        let global = Array1::from(vec![0.0]);

        // Create a draw position:
        // X O X
        // O X X
        // O X O
        spatial[[0, 0, 0]] = 1.0; // X
        spatial[[1, 0, 1]] = 1.0; // O
        spatial[[0, 0, 2]] = 1.0; // X

        spatial[[1, 1, 0]] = 1.0; // O
        spatial[[0, 1, 1]] = 1.0; // X
        spatial[[0, 1, 2]] = 1.0; // X

        spatial[[1, 2, 0]] = 1.0; // O
        spatial[[0, 2, 1]] = 1.0; // X
        spatial[[1, 2, 2]] = 1.0; // O

        assert!(game.is_terminal(&spatial.view(), &global.view()));
        assert_eq!(game.get_outcome(&spatial.view(), &global.view()), 0);
    }

    #[test]
    fn test_hash_different_for_different_states() {
        let game = TicTacToeGame::new();
        let (spatial1, global1) = TicTacToeGame::initial_state();

        let mut spatial2 = spatial1.clone();
        let global2 = global1.clone();
        spatial2[[0, 0, 0]] = 1.0;

        let hash1 = game.hash_state(&spatial1.view(), &global1.view());
        let hash2 = game.hash_state(&spatial2.view(), &global2.view());

        assert_ne!(hash1, hash2);
    }
}
