"""Exception hierarchy for ParseForge."""


class ParseForgeException(Exception):
    """Base exception for all ParseForge errors."""

    pass


class StrategyException(ParseForgeException):
    """Exception raised during strategy selection."""

    pass


class OCRError(ParseForgeException):
    """Exception raised during OCR processing."""

    pass


class LayoutError(ParseForgeException):
    """Exception raised during layout detection."""

    pass


class TableError(ParseForgeException):
    """Exception raised during table processing."""

    pass


class ParserError(ParseForgeException):
    """Exception raised during document parsing."""

    pass


class CheckpointError(ParseForgeException):
    """Exception raised during checkpoint operations."""

    pass


class ConfigurationError(ParseForgeException):
    """Exception raised due to configuration issues."""

    pass

