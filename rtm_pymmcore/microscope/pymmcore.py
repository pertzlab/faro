from rtm_pymmcore.microscope.base import AbstractMicroscope


class PyMMCoreMicroscope(AbstractMicroscope):
    """Intermediate base for all pymmcore-plus-based microscopes.

    Subclasses must set ``self.mmc`` to a ``CMMCorePlus`` instance.
    """

    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"

    def __init__(self):
        super().__init__()
        self.mmc = None  # subclasses must set this

    def validate_hardware(self, events) -> bool:
        if self.mmc is None:
            return True  # nothing to validate against
        from rtm_pymmcore.core.utils import validate_hardware
        return validate_hardware(events, self.mmc)
