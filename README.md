# OptionPricing

This project implements three methods to price **European options (Call and Put)** in Python:

1. **Black-Scholes Analytical Formula**
2. **Binomial Tree Method**
3. **Monte Carlo Simulation**

It allows comparison between numerical methods and the analytical solution, including **visualizations of convergence and accuracy**.

---

## 🔹 1. Black-Scholes Analytical Formula

The **Black-Scholes model** provides a closed-form solution for European option prices under the assumptions of:

* Constant volatility $\sigma$
* Constant risk-free rate $r$
* Log-normal distribution of the underlying asset

Formulas:

$$
\text{Call} = S \cdot N(d_1) - K e^{-rT} N(d_2)
$$

$$
\text{Put} = K e^{-rT} N(-d_2) - S \cdot N(-d_1)
$$

$$
d_1 = \frac{\ln(S/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
$$

* **Pros:** Fast, accurate, simple
* **Cons:** Limited to European options and specific assumptions

---

## 🔹 2. Binomial Tree Method

The **Binomial Tree** method estimates option prices by **modeling the underlying asset as moving up or down in discrete time steps**:

1. Construct a tree of possible asset prices over $N$ steps.
2. Compute option payoffs at the final nodes.
3. Recursively discount payoffs back to present.

* **Pros:** Works for American options, flexible
* **Cons:** Slower for high $N$, computationally intensive

**Visualization:** Option prices converge to Black-Scholes as $N$ increases.

---

## 🔹 3. Monte Carlo Simulation

Monte Carlo estimates option prices by **simulating many random paths of the underlying asset**:

$$
S_T = S \cdot \exp\big((r - 0.5\sigma^2)T + \sigma \sqrt{T} Z\big), \quad Z \sim N(0,1)
$$

Payoffs:

$$
\text{Call} = \max(S_T - K, 0), \quad \text{Put} = \max(K - S_T, 0)
$$

Option price is the **average of discounted payoffs**:

$$
\text{Option Price} \approx e^{-rT} \cdot \text{mean(payoffs)}
$$

* **Pros:** Flexible, handles complex options
* **Cons:** Requires many simulations, slower convergence

**Visualization:** Convergence of Monte Carlo prices to Black-Scholes as the number of simulations $M$ increases.

---

## 🔹 Comparison of Methods

| Method        | Accuracy          | Speed                         | Flexibility | Notes                                      |
| ------------- | ----------------- | ----------------------------- | ----------- | ------------------------------------------ |
| Black-Scholes | High (analytical) | Very fast                     | Low         | Only European options                      |
| Binomial Tree | Medium-High       | Moderate                      | Medium      | Works for American options                 |
| Monte Carlo   | Medium            | Slow (depends on simulations) | High        | Works for complex options (Asian, Barrier) |

---

## 🔹 Python Usage Example

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

# --- Black-Scholes ---
def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# --- Monte Carlo ---
M = 10000
Z = np.random.normal(size=M)
ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
call_mc = np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))
put_mc = np.exp(-r*T) * np.mean(np.maximum(K - ST, 0))

print("BS Call:", bs_price(S,K,T,r,sigma,"call"), "MC Call:", call_mc)
print("BS Put:", bs_price(S,K,T,r,sigma,"put"), "MC Put:", put_mc)
```

---

## 🔹 Features

* Price **European Call and Put options** using three different methods.
* Compare **numerical methods to analytical Black-Scholes**.
* Visualize **convergence and accuracy**.
* Flexible for **extension to complex option types**.
