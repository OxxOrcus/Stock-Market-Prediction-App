# Testing

This directory contains the test suite for the Stock Market Prediction App. The tests are built using the `pytest` framework and are designed to validate the core functionalities of the application.

## How to Run Tests

### 1. Install Dependencies

Before running the tests, ensure you have all the required dependencies installed. The testing framework requires `pytest`, `pytest-cov` for coverage reports, and `pytest-mock` for mocking external services.

You can install all necessary testing packages with the following command:

```bash
pip install pytest pytest-cov pytest-mock
```

### 2. Run the Test Suite

To run the entire test suite, navigate to the root directory of the project and execute the following command:

```bash
pytest
```

Pytest will automatically discover and run all the tests in the `tests` directory.

### 3. Generate a Coverage Report

To run the tests and see a coverage report, use the `--cov` flag. This will show you which parts of the `stock_predictor.py` module are covered by the tests.

```bash
pytest --cov=stock_predictor
```

This command will output a summary of the test coverage to the console. The current test suite provides **82% coverage** of the main application logic.
