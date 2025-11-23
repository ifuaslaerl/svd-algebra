import numpy as np
from svd_algebra import SVDTools, Visualizer

def compare_results(result_tuple, eps=1e-8):
    matriz_original = result_tuple[0][0]
    matriz_reconstructed = result_tuple[0][1]
    b_original = result_tuple[1][0]
    b_reconstructed = result_tuple[1][1]

    # Compare Matrices
    diff = matriz_original - matriz_reconstructed
    matrix_match = np.all(np.abs(diff) <= eps)

    # Compare Vectors (calculate distance)
    distancia_vetor = np.linalg.norm(b_reconstructed - b_original)

    return matrix_match, distancia_vetor

def main():
    distancias = []
    m_n_ratios = []
    ranks = []
    
    test_cases = 1000  # Reduced from 10000 for example speed
    limit_test = 50

    print(f"Running {test_cases} simulations...")

    for i in range(test_cases):
        # Generate random dimensions M, N and rank K
        M = np.random.randint(3, limit_test)
        N = np.random.randint(2, M) # M > N case, varies
        
        # Randomly decide M/N structure for variety
        if i % 3 == 0: # M = N
            N = M
        elif i % 3 == 1: # M < N
            N = np.random.randint(M, limit_test)
            
        K = np.random.randint(1, min(M, N) + 1)

        # Solve
        mat = SVDTools.generate_matrix(M, N, K)
        b = SVDTools.generate_vector(M)
        res = SVDTools.solve_linear_system(mat, b, verbose=False)
        
        # Compare
        match, dist = compare_results(res)
        
        distancias.append(dist)
        m_n_ratios.append(M/N)
        ranks.append(K/min(M, N))

    print("Plotting Norm vs M/N...")
    Visualizer.plot_scatter(m_n_ratios, distancias, 'Min Norm x M/N', 'M/N', 'Min Norm')

    print("Plotting Norm vs Rank Ratio...")
    Visualizer.plot_scatter(ranks, distancias, 'Min Norm x Rank/min(m,n)', 'Rank/min(m,n)', 'Min Norm')

if __name__ == "__main__":
    main()
