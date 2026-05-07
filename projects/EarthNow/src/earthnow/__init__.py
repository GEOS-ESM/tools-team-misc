# Setup package logging
import logging

logger = logging.getLogger(__name__)
# Add a NullHandler to prevent warnings when logging is not configured
logger.addHandler(logging.NullHandler())
