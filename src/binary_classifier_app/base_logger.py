import logging

BASE_MODULE_NAME = "binary_classifier_app"

BASE_LOGGER_SETTINGS = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "[%(levelname)s] | %(asctime)s | %(name)s | %(message)s"},
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
        }
    },
    "loggers": {
        BASE_MODULE_NAME: {"level": "DEBUG", "handlers": ["default"], "propagate": "no"}
    },
}

logging.config.dictConfig(BASE_LOGGER_SETTINGS)

BASE_LOGGER = logging.getLogger(BASE_MODULE_NAME)
