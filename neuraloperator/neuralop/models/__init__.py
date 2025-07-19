from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
from .wno import WNO, WNO1d, WNO2d, WNO3d
from .cno import CNO, CNO1d, CNO2d

# only import SFNO if torch_harmonics is built locally
try:
    from .sfno import SFNO
    from .local_no import LocalNO
except ModuleNotFoundError:
    pass
from .uno import UNO
from .uqno import UQNO
from .fnogno import FNOGNO
from .gino import GINO
from .base_model import get_model
