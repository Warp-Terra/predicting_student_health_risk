from src.data import prepare_data
from src.predict import predict_and_submit

def main():
    _, _, X_test, test_ids = prepare_data()
    predict_and_submit(X_test, test_ids, version="v1", use_cv=True)

if __name__ == "__main__":
    main()
