import numpy as np

def test(a=10, b=20, c=30):
    print(a, b, c)


def main():
    x= np.arange(1, 10).reshape(3, 3)
    y = np.arange(1, 7).reshape(2, 3)
    print(x)
    print(y)
    z = np.vstack([x, y])
    print(z)


if __name__ == "__main__":
    main()