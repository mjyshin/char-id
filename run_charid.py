from load_data import load_data


def main():
    data = load_data()
    print(data['Xtest'])


if __name__ == "__main__":
    main()
