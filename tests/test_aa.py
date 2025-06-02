

# TODO: write a proper test for this !
if __name__ == "__main__":
    # Test advantage alignment vectorized method
    beta = 0.7
    gamma = 0.9
    a1 = [3, 9, 4, 15]
    a2 = [14, 12, 2, 47]
    print(f"Element 2 should be {a1[1] + beta * gamma * (gamma**(1-0)*a1[0]) * a2[1]}")
    print(
        f"Element 3 should be {a1[2] + beta * gamma * (gamma**(2-0)*a1[0]+ gamma**(2-1)*a1[1]) * a2[2]}"
    )
    print(
        f"Element 4 should be {a1[3] + beta * gamma * (gamma**(3-0)*a1[0]+ gamma**(3-1)*a1[1]+ gamma**(3-2)*a1[2]) * a2[3]}"
    )
    print(
        advantages_to_aa_scores(np.array([a1]), np.array([a2]), beta=beta, gamma=gamma)
    )

    def readable_advantages_to_aa_scores(
        a1, a2, beta=1.0, gamma=0.9, regulate_var=False, time_decay=False
    ):
        """
        A more readable version of the advantage alignment calculation that shows step-by-step
        how the advantage alignment scores are computed.

        Args:
            a1 (list or np.ndarray): The first advantage array.
            a2 (list or np.ndarray): The second advantage array.
            beta (float, optional): The shaping factor. Defaults to 1.0.
            gamma (float, optional): The discount factor. Defaults to 0.9.
            regulate_var (bool, optional): Whether to regulate variance. Defaults to False.
            time_decay (bool, optional): Whether to apply 1/t regularization. Defaults to False.

        Returns:
            np.ndarray: The advantage alignment scores for each time step.
        """
        # Convert inputs to numpy arrays if they aren't already
        a1 = (
            np.array(a1, dtype=float)
            if not isinstance(a1, np.ndarray)
            else a1.astype(float)
        )
        a2 = (
            np.array(a2, dtype=float)
            if not isinstance(a2, np.ndarray)
            else a2.astype(float)
        )

        # Handle both 1D arrays and 2D arrays (where rows are different matches)
        single_match = a1.ndim == 1
        if single_match:
            a1 = a1.reshape(1, -1)
            a2 = a2.reshape(1, -1)

        T = a1.shape[1]  # Number of time steps
        aa_scores = np.zeros_like(a1)

        # For clarity, process one match at a time
        for match_idx in range(a1.shape[0]):
            match_a1 = a1[match_idx]
            match_a2 = a2[match_idx]

            for t in range(T):
                # Calculate the sum of discounted past advantages
                discounted_sum = 0
                for k in range(t):
                    discounted_sum += (gamma ** (t - k)) * match_a1[k]

                # Calculate the alignment term
                alignment_term = beta * gamma * discounted_sum * match_a2[t]

                # Apply time decay if specified
                if time_decay:
                    alignment_term = alignment_term / (t + 1)

                # Calculate the final score
                aa_scores[match_idx, t] = match_a1[t] + alignment_term

        # If variance regulation is needed, apply it after all calculations
        if regulate_var:
            # Get the last advantage values and alignment terms for each match
            last_idx = T - 1
            a1_last = a1[:, last_idx]
            alignment_terms_last = aa_scores[:, last_idx] - a1[:, last_idx]

            # Calculate standard deviations
            std_a1 = np.std(a1_last)
            std_alignment = np.std(alignment_terms_last) + 1e-10

            # Calculate regulation coefficient
            reg_coef = std_a1 / std_alignment

            # Apply the regulation to all alignment terms
            for match_idx in range(a1.shape[0]):
                for t in range(T):
                    alignment_term = aa_scores[match_idx, t] - a1[match_idx, t]
                    aa_scores[match_idx, t] = a1[match_idx, t] + (
                        alignment_term * reg_coef
                    )

        # If input was 1D, return a 1D array
        if single_match:
            return aa_scores[0]

        return aa_scores

    # Test the readable version and compare with vectorized version
    print("\nComparing readable vs vectorized implementation:")
    vectorized_result = advantages_to_aa_scores(
        np.array([a1]), np.array([a2]), beta=beta, gamma=gamma
    )[0]
    readable_result = readable_advantages_to_aa_scores(a1, a2, beta=beta, gamma=gamma)

    print(f"Readable implementation result: {readable_result}")
    print(f"Vectorized implementation result: {vectorized_result}")
    print(f"Match? {np.allclose(readable_result, vectorized_result)}")

    # Test with a batch of advantages
    print("\nTesting with multiple matches:")
    batch_a1 = np.array([[3, 9, 4, 15], [2, 5, 8, 3]])
    batch_a2 = np.array([[14, 12, 2, 47], [10, 7, 9, 5]])

    batch_vectorized = advantages_to_aa_scores(
        batch_a1, batch_a2, beta=beta, gamma=gamma
    )
    batch_readable = readable_advantages_to_aa_scores(
        batch_a1, batch_a2, beta=beta, gamma=gamma
    )

    print(f"Readable batch result shape: {batch_readable}")
    print(f"Vectorized batch result shape: {batch_vectorized}")
    print(f"Batch match? {np.allclose(batch_readable, batch_vectorized)}")

    # Test rewards_to_rloo_advantages with a simple 3x3 array
    print("\nTesting rewards_to_rloo_advantages with a 3x3 array:")
    test_rewards = np.array(
        [
            [1, 2, 3],  # Match 1 rewards
            [4, 5, 6],  # Match 2 rewards
            [7, 8, 9],  # Match 3 rewards
        ]
    )
    discount_factor = 0.9

    # Calculate expected result manually
    # First calculate discounted rewards to go
    expected_rtg = np.array(
        [
            [1 + 0.9 * 2 + 0.9**2 * 3, 2 + 0.9 * 3, 3],
            [4 + 0.9 * 5 + 0.9**2 * 6, 5 + 0.9 * 6, 6],
            [7 + 0.9 * 8 + 0.9**2 * 9, 8 + 0.9 * 9, 9],
        ]
    )

    # Then calculate leave-one-out advantage
    expected_rloo = np.zeros_like(expected_rtg)
    for i in range(3):
        for j in range(3):
            others_avg = (
                np.sum(expected_rtg[:, j]) - expected_rtg[i, j]
            ) / 2  # n-1 = 2
            expected_rloo[i, j] = expected_rtg[i, j] - others_avg

    rloo_result = rewards_to_rloo_advantages(test_rewards, discount_factor)

    print(f"Test rewards:\n{test_rewards}")
    print(f"Discounted rewards-to-go:\n{expected_rtg}")
    print(f"Expected RLOO advantages:\n{expected_rloo}")
    print(f"Function output:\n{rloo_result}")
    print(f"Match? {np.allclose(expected_rloo, rloo_result)}")
    import time
    from contextlib import contextmanager

    @contextmanager
    def time_it(label="Time taken"):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            print(f"{label}: {end - start:.4f} seconds")

    nb_games = 10
    game_lengths = 8
    a1 = np.random.random((nb_games, game_lengths))
    a2 = np.random.random((nb_games, game_lengths))
    with time_it("Readable"):
        s = readable_advantages_to_aa_scores(a1, a2)
        print(s)

    with time_it("Chad"):
        s = advantages_to_aa_scores(a1, a2)
        print(s)
    print("Done")