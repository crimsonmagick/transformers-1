import logging
import time

logger = logging.getLogger(__name__)


def profile(operation, *args, **kwargs):
    operation_name = kwargs.get('operation_name') if kwargs.get('operation_name') is not None \
        else operation.__name__
    before_ts_ms = int(time.time() * 1000)
    output = operation(*args, **kwargs)
    after_ts_ms = int(time.time() * 1000)
    logger.info(f"{operation_name}_execution_time={after_ts_ms - before_ts_ms} ms")
    return output
