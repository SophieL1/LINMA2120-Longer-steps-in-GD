import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.optimize import fsolve

from data_processing import get_data_matrices, get_Lipschitz_constant

global A, b
global X, y

def f(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2

def grad_f(x):
    return A.T @ (A @ x - b)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def f_logi(theta):
    """
    Compute the function value of the logistic regression function at theta.
    """
    m = X.shape[0]
    h = sigmoid(X @ theta)
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def grad_f_logi(theta):
    """
    Compute the gradient of f(theta) for logistic functin.
    """
    m = X.shape[0] 
    h = sigmoid(X @ theta)
    return (1 / m) * (X.T @ (h - y))

def f_logi_plus_lin(theta, grad_xstar):
    """
    Compute the function value f(theta) - <grad_x0, theta>
    of the logistic regression function translated of a linear function, 
    so as to have 0 gradient at x_star.
    """
    m = X.shape[0]
    h = sigmoid(X @ theta)
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) - grad_xstar @ theta

def grad_f_logi_plus_lin(theta, grad_xstar):
    """
    Compute the gradient of f(theta) - <grad_x0, theta>
    of the logistic regression function translated of a linear function,
    so as to have 0 gradient at x_star.
    """
    m = X.shape[0] 
    h = sigmoid(X @ theta)
    return (1 / m) * (X.T @ (h - y)) - grad_xstar

def get_true_solution(X, y):
    """
    Find an approximate true solution to the logistic regression problem
    """
    model = LogisticRegression(fit_intercept=False, solver='newton-cg', max_iter=100000)
    model.fit(X, y)
    theta_star = np.hstack([model.coef_.flatten()])
    # Compute the minimum objective value f*
    f_star = f_logi(theta_star)
    return theta_star, f_star

# Define the optimization algorithms
# Gradient descent with constant step size
def gradient_descent_constant(fun, grad_fun, x0, f_star, step_size, max_iter):
    x = x0
    errors = np.zeros(max_iter)
    error_min = fun(x) - f_star
    for i in range(max_iter):
        errors[i] = fun(x) - f_star
        if errors[i] > error_min:
            errors[i] = error_min
        else :
            error_min = errors[i]
        x = x - step_size * grad_fun(x)
    return x, errors

# Gradient descent with dynamic step size (non-constant step size)
def gradient_descent_dynamic(fun, grad_fun, x0, f_star, step_sizes, max_iter):
    x = x0
    errors = np.zeros(max_iter)
    error_min = fun(x) - f_star
    for i in range(max_iter):
        errors[i] = fun(x) - f_star
        if errors[i] > error_min:
            errors[i] = error_min
        else :
            error_min = errors[i]
        x = x - step_sizes[i] * grad_fun(x)    
    return x, errors

# Nesterov's accelerated gradient descent
def nesterov_accelerated_gradient_descent(fun, grad_fun, x0, f_star, max_iter, L):
    x = x0
    y = x0
    step_size = 1 / L
    t = 1
    t_next = 1
    errors = np.zeros(max_iter)
    for i in range(max_iter):
        errors[i] = fun(x) - f_star
        x_next = y - step_size * grad_fun(y)
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_next + (t - 1) / t_next * (x_next - x)
        x = x_next
        t = t_next
    return x, errors

# Compute specific stepsizes / schedules
# Taylor, Hendricks and Glineur's optimal constant step size
def get_optimal_constant_step_size(N, L):
    def equation(h):
        return 1 / (2 * N * h + 1) - (1 - h) ** (2 * N)
    
    # Initial guess for h_opt (start in the middle of the [0, 1] range)
    initial_guess = 0.5
    
    # Solve the equation using fsolve
    h_opt = fsolve(equation, initial_guess, xtol=1e-10)
    
    return h_opt/L

# T-V dynamic step size (Teboulle-Vaisbourd)
def get_Vaisbourd_Teboulle_step_size(N, L):
    h = np.zeros(N)
    h[0] = np.sqrt(2)
    T = h[0]/L
    for i in range(1, N):
        h[i] = (-L*T + np.sqrt((L*T)**2 + 8*(L*T+1)))/2
        T += h[i]/L

    return h/L

# 5. Alternating dynamic step size mentionned by Das Gupta
def get_alternating_step_size(N, L):
    h = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            h[i] = 2.9
        else:
            h[i] = 1.5

    return h / L 

# Das Gupta dynamic step size (expensive computation, so the first 50 steps are precomputed and storedon their github repo)
def get_das_gupta_step_size50(L):
    # Taken from https://github.com/Shuvomoy/BnB-PEP-code/blob/main/Misc/stpszs.jl
    h_50 = [1.5958743518790774, 1.4203770234563378, 2.971211184884086, 1.4157274281846162, 1.995741454373446, 
                1.4152169068658134, 8.549741557142763, 1.4154688479452724, 1.9983336889867038, 1.4147088064618587, 
                4.888232118878028, 1.6005594184145655, 1.4161180326078795, 3.3605483314412066, 1.4158245523573785, 
                2.1171962361361203, 1.4145676908196327, 2.320338004765043, 1.4160303079882408, 4.907259290717155, 
                1.4329242645055837, 1.964171439142215, 1.4130061410783987, 36.96886074199298, 1.415489557404075, 
                1.9996059863093698, 1.4145244241336101, 3.7268306084082283, 1.5268410248149245, 1.4782602939339355, 
                2.2558071881603117, 1.414347161790353, 13.474201968821102, 1.5976750370133366, 1.4188627098610425, 
                2.9643322594240185, 1.4230920853398292, 1.9895190521714607, 1.4128709461202384, 6.6622453348451005, 
                1.514751947716923, 1.4865992890480064, 3.279911713894502, 1.4161590010827143, 2.055739999107617, 
                1.4168372776764715, 2.4219460576308194, 1.4147756009008488, 8.175336727653681, 1.500403734613415] 
    return np.array(h_50) / L

# Grimmer's pattern of size 31
def get_grimmer_step_size31(N, L):
    # Taken from Grimmer et al. Accelerated Gradient Descent via Long Steps, 2023
    h31 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 72.3,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4]
    extended_h31 = 40*h31
    extended_h31 = np.array(extended_h31)[0:N]
    return extended_h31 / L

# Grimmer's pattern of size 63
def get_grimmer_step_size63(N, L):
    # Taken from Grimmer et al. Accelerated Gradient Descent via Long Steps, 2023
    h63 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 14.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 164.0,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 14.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4]

# Grimmer's pattern of size 127
def get_grimmer_step_size127(N, L):
    # Taken from Grimmer et al. Accelerated Gradient Descent via Long Steps, 2023
    h127 = [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 23.5,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 370.0,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 23.5,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2,
    1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4]
    extended_h127 = 16*h127
    extended_h127 = np.array(extended_h127)[0:N]
    return extended_h127 / L

# Silver steps by Altschuler and Parrilo
def get_silver_step_size(n, L):
    rho = 1 + np.sqrt(2)

    def recursive_schedule(k):
        if k == 0:
            return [np.sqrt(2)]
        prev_schedule = recursive_schedule(k - 1)
        return prev_schedule + [1 + rho ** (k - 1)] + prev_schedule
    
    # Find smallest k st 2^k - 1 >= n (to get the full schedule, truncated afterwards i needed)
    k = 0
    while (2 ** k - 1) < n:
        k += 1
    full_schedule = recursive_schedule(k)
    silversteps = np.array(full_schedule[0:n])
    
    return silversteps / L

# Example on logistic regression problem
def logistic_regression():
    global X, y
    translated = True

    np.random.seed(1)
    X_train, X_test, y_train, y_test = get_data_matrices()
    X = X_train
    y = y_train.to_numpy().flatten()
    L = get_Lipschitz_constant(X_train)
    ndim = X_train.shape[1]
    x0 = np.random.rand(ndim)

    if translated:
        x_star = np.ones(ndim)
        grad_x_star = grad_f_logi(x_star)
        f_star = f_logi_plus_lin(x_star, grad_x_star)
        grad_zero = grad_f_logi_plus_lin(x_star, grad_x_star)
        print(f"Gradient norm at x_star: {np.linalg.norm(grad_zero):.5f}")
        fun = lambda theta: f_logi_plus_lin(theta, grad_x_star)
        grad_fun = lambda theta: grad_f_logi_plus_lin(theta, grad_x_star)

    else:    
        x_star, f_star = get_true_solution(X_train, y_train)
        fun = f_logi
        grad_fun = grad_f_logi

    X = X_train
    step_size = 1 / L
    max_iter = 50
    # 1. Classic gradient descent
    _, errors_classic = gradient_descent_constant(fun, grad_fun, x0, f_star, step_size, max_iter)

    # 2. THG optimal constant stepsize
    step_size_opt = get_optimal_constant_step_size(max_iter, L)
    _, errors_optimal = gradient_descent_constant(fun, grad_fun, x0, f_star, step_size_opt, max_iter)

    # 3. Vaisbourd-Teboulle dynamic step size
    step_sizes_VT = get_Vaisbourd_Teboulle_step_size(max_iter, L)
    _, errors_VT = gradient_descent_dynamic(fun, grad_fun, x0, f_star, step_sizes_VT, max_iter)
    
    # 4. Alternating dynamic step size mentionned by Das Gupta
    step_sizes_das_gupta = get_das_gupta_step_size50(L)
    _, errors_das_gupta = gradient_descent_dynamic(fun, grad_fun, x0, f_star, step_sizes_das_gupta, 50)

    # 5. Grimmer's pattern of size 31
    step_size_Grimmer31 = get_grimmer_step_size31(max_iter, L)
    _, errors_Grimmer31 = gradient_descent_dynamic(fun, grad_fun, x0, f_star, step_size_Grimmer31, max_iter)
    
    # 6. Silver steps of size 31
    step_size_Silver = get_silver_step_size(max_iter, L)
    _, errors_Silver = gradient_descent_dynamic(fun, grad_fun, x0, f_star, step_size_Silver, max_iter)

    _, errors_nesterov = nesterov_accelerated_gradient_descent(fun, grad_fun, x0, f_star, max_iter, L)

    # plot the results
    plt.plot(np.arange(max_iter), errors_classic, label="Constant stepsize 1")
    plt.plot(np.arange(max_iter), errors_optimal, label="Optimal constant stepsize $h_{\mathrm{opt}}$")
    plt.plot(np.arange(max_iter), errors_VT, label="Vaisbourd-Teboulle dynamic stepsize")
    plt.plot(np.arange(50), errors_das_gupta, label="Das Gupta dynamic stepsize")
    plt.plot(np.arange(max_iter), errors_Grimmer31, label="Grimmer's pattern of size 31")
    plt.plot(np.arange(max_iter), errors_Silver, label="Silver steps schedule")
    plt.plot(np.arange(max_iter), errors_nesterov, label="Nesterov's accelerated gradient descent", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig('logistic_regression.pdf')
    plt.show()

# Example on linear system solving problem
def linear_system_solving():
    global A, b

    ndim = 20
    np.random.seed(1)
    A = np.random.rand(ndim, ndim)
    x = np.random.rand(ndim)
    b = A @ x
    x0 = np.random.rand(ndim)
    f_star = f(x)
    L = np.linalg.norm(A.T @ A, 2)
    step_size = 1 / L
    max_iter = 50
    # 1. Classic gradient descent
    _, errors_classic = gradient_descent_constant(f, grad_f, x0, f_star, step_size, max_iter)

    # 2. THG optimal constant stepsize
    step_size = get_optimal_constant_step_size(max_iter, L)
    _, errors_optimal = gradient_descent_constant(f, grad_f, x0, f_star, step_size, max_iter)

    # 3. Vaisbourd-Teboulle dynamic step size
    step_sizes_VT = get_Vaisbourd_Teboulle_step_size(max_iter, L)
    _, errors_VT = gradient_descent_dynamic(f, grad_f, x0, f_star, step_sizes_VT, max_iter)
    
    # 4. Alternating dynamic step size mentionned by Das Gupta
    step_sizes_das_gupta = get_das_gupta_step_size50(L)
    _, errors_das_gupta = gradient_descent_dynamic(f, grad_f, x0, f_star, step_sizes_das_gupta, 50)

    # 5. Grimmer's pattern of size 31
    step_size_Grimmer31 = get_grimmer_step_size31(max_iter, L)
    _, errors_Grimmer31 = gradient_descent_dynamic(f, grad_f, x0, f_star, step_size_Grimmer31, max_iter)

    # 6. Silver steps of size 31
    step_size_Silver = get_silver_step_size(max_iter, L)
    _, errors_Silver = gradient_descent_dynamic(f, grad_f, x0, f_star, step_size_Silver, max_iter)
        
    _, errors_nesterov = nesterov_accelerated_gradient_descent(f, grad_f, x0, f_star, max_iter, L)

    # plot the results
    plt.plot(np.arange(max_iter), errors_classic, label="Constant stepsize 1")
    plt.plot(np.arange(max_iter), errors_optimal, label="Optimal constant stepsize $h_{\mathrm{opt}}$")
    plt.plot(np.arange(max_iter), errors_VT, label="Vaisbourd-Teboulle dynamic stepsize")
    plt.plot(np.arange(max_iter), errors_das_gupta, label="Das Gupta dynamic stepsize")
    plt.plot(np.arange(max_iter), errors_Grimmer31, label="Grimmer's pattern of size 31")
    plt.plot(np.arange(max_iter), errors_Silver, label="Silver steps schedule")
    plt.plot(np.arange(max_iter), errors_nesterov, label="Nesterov's accelerated gradient descent", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.yscale("log")

    plt.legend()
    plt.grid()
    plt.savefig('linear_system_solving.pdf')
    plt.show()

def plot_theoretical_rates():
    # plot theoretical results
    max_iter = 1024
    L = 1
    x0 = 1
    x_star = 0

    # 1. Classic gradient descent
    th_errors_classic = np.array([np.linalg.norm(x0-x_star)**2*L*1/(4*i+2) for i in range(max_iter)])
    plt.plot(np.arange(max_iter), th_errors_classic, label="Constant stepsize 1")

    # 2. THG optimal constant stepsize
    step_size_opt = np.zeros(max_iter)
    for i in range(1, max_iter+1):
        step_size_opt[i-1] = get_optimal_constant_step_size(i, L)*L

    # Check with number presented in Teboulle23 Table 1 page 81 for max_iter=100, get 395.10932941232727 :)
    #print(1/max(1/(2*max_iter*step_size_opt[-1] + 1), (1-step_size_opt[-1])**(2*max_iter))) 
    th_errors_optimal = np.array([np.linalg.norm(x0-x_star)**2*L/2*max(1/(2*i*step_size_opt[i-1] + 1), (1-step_size_opt[i-1])**(2*i)) for i in range(1, max_iter+1)])
    plt.plot(np.arange(max_iter), th_errors_optimal, label="Optimal constant stepsize $h_{\mathrm{opt}}$")

    # 3. Vaisbourd-Teboulle dynamic step size
    step_sizes_VT = get_Vaisbourd_Teboulle_step_size(max_iter, L)
    Ts_VT = np.concatenate(([0], np.cumsum(step_sizes_VT)))
    # Check with number presented in Teboulle23 Table 1 page 81 for max_iter=100, get 391.66104219 :)
    #print((2*L*Ts_VT[-1] + 1))
    th_errors_VT = np.array([np.linalg.norm(x0-x_star)**2*L*1/(2*(2*L*Ts_VT[i] + 1)) for i in range(max_iter)])
    plt.plot(np.arange(max_iter), th_errors_VT, label="Vaisbourd-Teboulle dynamic stepsize")

    # 4. Das Gupta optimized step size
    th_errors_das_gupta = np.array([np.linalg.norm(x0-x_star)**2*L*0.156/i**1.178 for i in range(50)])
    plt.plot(np.arange(50), th_errors_das_gupta, label="Das Gupta dynamic stepsize", linestyle="--")

    # 5. Grimmer's pattern of size 31
    step_size_Grimmer31 = get_grimmer_step_size31(31, L)*L
    avg_Grimmer31 = np.mean(step_size_Grimmer31)
    indices = np.arange(31, max_iter + 1, 31)
    th_errors_Grimmer31 = np.array([np.linalg.norm(x0 - x_star)**2*L*1/(avg_Grimmer31*i) for i in indices])
    plt.scatter(indices, th_errors_Grimmer31, label="Grimmer's pattern of size 31", marker="x", color="purple")

    # 6. Silver steps by Altschuler and Parrilo
    indices = [2**k - 1 for k in range(1, int(np.log2(max_iter)) + 1) if 2**k - 1 < max_iter]
    th_errors_Silver = np.array([np.linalg.norm(x0 - x_star)**2*L* 1/(2*i**(np.log2(np.sqrt(2)+1))) for i in indices])
    plt.scatter(indices, th_errors_Silver, label="Silver steps schedule", marker="x", color="red")
    
    plt.xlabel("Iteration")
    plt.ylabel("Theoretical error upper bound")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig('theoretical_rates.pdf')
    plt.show()
    

if __name__ == "__main__" :
    plot_theoretical_rates()
    logistic_regression()
    linear_system_solving()