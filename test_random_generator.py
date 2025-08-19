import numpy as np
from random_generator import sample_h_q_w_from_config


def main():
    data = sample_h_q_w_from_config("config.yaml")

    q = data["q"]
    T = data["T"]
    W = data["W"]
    h = data["h"]

    print("Shapes:")
    print("  q:", q.shape)
    print("  T:", T.shape)
    print("  W:", W.shape)
    print("  h:", h.shape)

    print("\nFirst 3 rows of q:")
    print(q[:3])
    print("\nT matrix:")
    print(T)
    print("\nFirst 3 rows of W:")
    print(W[:3])
    print("\nFirst 10 entries of h:")
    print(h[:10])


if __name__ == "__main__":
    main()


