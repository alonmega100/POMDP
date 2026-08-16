AVAILABLE_SENSORS = {
    "line_3": {
        "class": "LineSensor",
        "params": {"num_blocks": 3},
        "display_name": "Line Sensor (3 blocks)",
    },
    "line_4": {
        "class": "LineSensor",
        "params": {"num_blocks": 4},
        "display_name": "Line Sensor (4 blocks)",
    },
    "cone_3": {
        "class": "ConeSensor",
        "params": {"depth": 3},
        "display_name": "Cone Sensor (depth 3)",
    },
    "cone_4": {
        "class": "ConeSensor",
        "params": {"depth": 4},
        "display_name": "Cone Sensor (depth 4)",
    },
    "front_row_3_1": {
        "class": "FrontRowSensor",
        "params": {"width": 3, "distance": 1},
        "display_name": "Front Row / Pawn (W3, D1)",
    },
    "front_row_5_1": {
        "class": "FrontRowSensor",
        "params": {"width": 5, "distance": 1},
        "display_name": "Front Row (W5, D1)",
    },
    "front_row_3_2": {
        "class": "FrontRowSensor",
        "params": {"width": 3, "distance": 2},
        "display_name": "Front Row (W3, D2)",
    },
    "front_row_5_2": {
        "class": "FrontRowSensor",
        "params": {"width": 5, "distance": 2},
        "display_name": "Front Row (W5, D2)",
    },
    "diagonal_3": {
        "class": "DiagonalSensor",
        "params": {"num_blocks": 3},
        "display_name": "Diagonal Sensor (3 blocks)",
    },
    "diagonal_4": {
        "class": "DiagonalSensor",
        "params": {"num_blocks": 4},
        "display_name": "Diagonal Sensor (4 blocks)",
    },
}

ACTIVE_SENSOR = "diagonal_3"
# MODELS_TO_TRAIN = ["front_row_3_1", "front_row_5_1", "front_row_3_2", "front_row_5_2", "diagonal_3", "diagonal_4"]
MODELS_TO_TRAIN = ["diagonal_3"]
# sensor names for me = ["line_3", "line_4", "cone_3", "cone_4", "front_row_3_1", "front_row_5_1", "front_row_3_2", "front_row_5_2", "diagonal_3", "diagonal_4"]


SAVED_MODELS_DIR = "saved_models"
NUMBER_OF_ENVIRONMENTS = 1000
NUM_EPISODES_FOR_AGENT_RUN = 5
WINDOW_SIZE = 50000
TOTAL_TRAINING_STEPS = 100000
# TOTAL_TRAINING_STEPS = 10

TRAIN_ON_HARD_ENVS_ONLY = True
PLAY_ON_HARD_ENVS_ONLY = True