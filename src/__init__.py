import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.as_posix()
if ROOT_DIR not in sys.path:
    # add package directory in system path, so submodules can easily be accessed
    sys.path.append(ROOT_DIR) 
    
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)
logger = logging.getLogger(__name__)