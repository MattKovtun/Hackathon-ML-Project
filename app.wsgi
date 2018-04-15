import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/ai-spring/")

from server import app as application
application.secret_key = 'ai-spring'