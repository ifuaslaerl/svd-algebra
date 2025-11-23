import numpy as np
import matplotlib.pyplot as plt

class SVDTools:
    """
    A collection of tools for Singular Value Decomposition (SVD) analysis.
    """

    @staticmethod
    def generate_matrix(m: int, n: int, k: int) -> np.ndarray:
        """
        Generates a random matrix with M lines, N columns, and a specific rank K.
        Formula: A = sum(u * v^T) for i=1 to k
        """
        matriz = np.zeros((m, n))
        for i in range(k):
            u = np.random.randint(-9, 9, (m, 1))
            v = np.random.randint(-9, 9, (1, n))
            matriz += u * v
        return matriz

    @staticmethod
    def generate_vector(n: int) -> np.ndarray:
        """Generates a random vector of size n."""
        return np.random.randint(-9, 9, n)

    @staticmethod
    def get_rank_and_s_matrix(A_diag: np.ndarray, m: int, n: int):
        """
        Calculates rank and constructs the S matrix (diagonal matrix of singular values).
        """
        rank = min(m, n)
        # A_diag is a list of roots of eigenvalues in descending order
        A_list = A_diag.copy()
        
        for i in range(len(A_list)):
            if A_list[i] < 1e-12:  # Precision check
                A_list[i:] = 0
                rank = i
                break

        S = np.zeros((m, n))
        S[:min(m, n), :min(m, n)] = np.diag(A_list[:min(m, n)])
        return rank, S

    @staticmethod
    def solve_linear_system(matriz: np.ndarray, b: np.ndarray, verbose: bool = True, decimals: int = 5):
        """
        Solves Ax = b using SVD decomposition.
        Returns ((Original Matrix, Reconstructed Matrix), (b, Reconstructed b)).
        """
        m, n = matriz.shape

        if verbose:
            print(f'Matrix ({m}x{n}):\n{matriz}')
            print(f'\nVector b:\n{b}')

        U, A_vals, Vt = np.linalg.svd(matriz)
        rank, S = SVDTools.get_rank_and_s_matrix(A_vals, m, n)
        
        matriz_temp = U @ S @ Vt

        if verbose:
            print('\nOrthonormal Matrix U:\n', np.round(U, decimals))
            print('\nDiagonal Matrix S:\n', np.round(S, decimals))
            print('\nOrthonormal Matrix V (transposed):\n', np.round(Vt, decimals))
            print(f'\nRank: {rank} (Full rank: {min(m,n)})')

        # Calculate Pseudo-inverse of S
        A_inv = []
        for i in A_vals:
            if i == 0: break
            A_inv.append(1/i)
        
        S_inv = np.zeros((n, m))
        for i in range(min(m, n, rank)):
            S_inv[i][i] = A_inv[i]

        # Calculate x = V * S_inv * U^T * b
        x = Vt.T @ S_inv @ U.T @ b

        if verbose:
            print(f'\nVector x (Ax ~ b):\n{np.round(x, decimals)}')

        b_temp = matriz @ x
        return ((matriz, matriz_temp), (b, b_temp))

    @staticmethod
    def reduced_rank_approximation(matriz: np.ndarray, k: int) -> np.ndarray:
        """
        Approximates a matrix using only the first k singular values.
        """
        U, S, Vt = np.linalg.svd(matriz, full_matrices=False)
        
        # Construct approximation
        Ak = np.zeros(matriz.shape)
        for i in range(k):
            # S[i] is the singular value sigma_i
            # We use outer product of i-th columns of U and V
            Ak += S[i] * np.outer(U[:, i], Vt[i, :])
            
        return Ak

    @staticmethod
    def frobenius_norm(singular_values: np.ndarray, k: int) -> float:
        """
        Calculates Frobenius norm distance based on omitted singular values.
        """
        # Ideally this takes the remaining singular values, but following
        # the report logic which passes 'A' (singular values list):
        soma = 0
        for i in range(k, len(singular_values)):
            soma += singular_values[i]**2
        return soma**0.5

class Visualizer:
    """
    Helper class for plotting SVD results.
    """
    @staticmethod
    def plot_scatter(x, y, title, xlabel, ylabel):
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel, color='black')
        plt.ylabel(ylabel, color='black')
        plt.grid(False)
        plt.scatter(x, y, color='black')
        plt.show()

    @staticmethod
    def show_image(image):
        if len(image.shape) == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def rgb_to_gray(image):
        return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
