PREDICTION_CONFIG = {
    "runtime_in_minutes": 10 * 60,         # Prediction will stop either after specified runtime or after reaching
    "max_number_of_results": 1000,     # max number of results, whichever comes first.
    "create_plots": False           # Create a plot for every example containing different predictions and events
}

DATA_CONFIG = {
    "database_path": 'data/tinydb/dbtest2.json',
    "csv_input_path": "data/csv/data_18.csv"
}

