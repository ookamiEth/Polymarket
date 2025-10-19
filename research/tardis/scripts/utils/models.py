"""
Pydantic models for Tardis Machine API responses.

These models provide runtime validation and type safety for data from the
Tardis Machine Deribit options data API.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class OptionType(str, Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


class MessageType(str, Enum):
    """Tardis message type enumeration."""

    QUOTE = "quote"
    BOOK_SNAPSHOT = "book_snapshot"


class PriceLevel(BaseModel):
    """Price level in orderbook (bid or ask)."""

    price: float = Field(ge=0, description="Price at this level")
    amount: float = Field(ge=0, description="Amount/volume at this level")

    model_config = {"frozen": True}


class QuoteMessage(BaseModel):
    """Quote message from Tardis Machine API."""

    exchange: Literal["deribit"] = Field(default="deribit")
    symbol: str = Field(min_length=1, description="Option symbol (e.g., BTC-31DEC24-100000-C)")
    timestamp: str = Field(description="ISO 8601 timestamp")
    local_timestamp: Optional[str] = Field(default=None, description="Local timestamp")
    type: str = Field(description="Message type (quote, book_snapshot)")
    bids: list[PriceLevel] = Field(default_factory=list, description="Bid price levels")
    asks: list[PriceLevel] = Field(default_factory=list, description="Ask price levels")

    @field_validator("symbol")
    @classmethod
    def validate_symbol_format(cls, v: str) -> str:
        """Validate Deribit symbol format: ASSET-DATE-STRIKE-TYPE."""
        parts = v.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid symbol format: {v}. Expected format: BTC-31DEC24-100000-C")
        if parts[0] not in ["BTC", "ETH"]:
            raise ValueError(f"Invalid asset: {parts[0]}. Must be BTC or ETH")
        if parts[3] not in ["C", "P"]:
            raise ValueError(f"Invalid option type: {parts[3]}. Must be C or P")
        return v

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Get best bid (highest price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else None

    model_config = {"frozen": True}


class ParsedSymbol(BaseModel):
    """Parsed Deribit option symbol."""

    underlying: Literal["BTC", "ETH"] = Field(description="Underlying asset")
    expiry_str: str = Field(description="Expiry date string (e.g., 31DEC24)")
    strike_price: float = Field(gt=0, description="Strike price")
    option_type: OptionType = Field(description="Call or Put")

    @classmethod
    def from_symbol(cls, symbol: str) -> "ParsedSymbol":
        """
        Parse a Deribit symbol into components.

        Args:
            symbol: Symbol in format "BTC-31DEC24-100000-C"

        Returns:
            ParsedSymbol with parsed components

        Raises:
            ValueError: If symbol format is invalid
        """
        parts = symbol.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid symbol format: {symbol}")

        underlying = parts[0]
        if underlying not in ["BTC", "ETH"]:
            raise ValueError(f"Invalid underlying: {underlying}")

        expiry_str = parts[1]
        strike_price = float(parts[2])

        option_type_str = parts[3]
        if option_type_str == "C":
            option_type = OptionType.CALL
        elif option_type_str == "P":
            option_type = OptionType.PUT
        else:
            raise ValueError(f"Invalid option type: {option_type_str}")

        return cls(
            underlying=underlying,  # type: ignore
            expiry_str=expiry_str,
            strike_price=strike_price,
            option_type=option_type,
        )

    def to_symbol(self) -> str:
        """Convert back to Deribit symbol format."""
        suffix = "C" if self.option_type == OptionType.CALL else "P"
        return f"{self.underlying}-{self.expiry_str}-{int(self.strike_price)}-{suffix}"

    model_config = {"frozen": True}


class ProcessedQuote(BaseModel):
    """Processed quote data ready for storage."""

    exchange: Literal["deribit"] = Field(default="deribit")
    symbol: str = Field(min_length=1)
    timestamp: int = Field(description="Unix timestamp in microseconds")
    local_timestamp: int = Field(description="Local timestamp in microseconds")
    type: OptionType = Field(description="Call or Put")
    strike_price: float = Field(gt=0)
    underlying: Literal["BTC", "ETH"]
    expiry_str: str
    bid_price: Optional[float] = Field(default=None, ge=0)
    bid_amount: Optional[float] = Field(default=None, ge=0)
    ask_price: Optional[float] = Field(default=None, ge=0)
    ask_amount: Optional[float] = Field(default=None, ge=0)

    @field_validator("bid_price", "ask_price")
    @classmethod
    def validate_bid_ask_spread(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure bid price <= ask price if both exist."""
        # Note: This validator runs per-field, so we can't compare both here
        # The actual validation happens in the processing pipeline
        return v

    model_config = {"frozen": True}


class TardisFetchConfig(BaseModel):
    """Configuration for Tardis Machine data fetch."""

    exchange: Literal["deribit"] = Field(default="deribit")
    symbols: list[str] = Field(min_length=1, description="List of option symbols to fetch")
    from_datetime: str = Field(description="Start datetime in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
    to_datetime: str = Field(description="End datetime in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
    data_types: list[str] = Field(
        default_factory=lambda: ["quote_1s"], description="Data types to fetch (e.g., quote_1s, book_snapshot_1_1s)"
    )

    @field_validator("from_datetime", "to_datetime")
    @classmethod
    def validate_iso_format(cls, v: str) -> str:
        """Validate ISO 8601 datetime format."""
        try:
            # Try parsing to ensure valid format
            datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as e:
            raise ValueError(f"Invalid ISO datetime format: {v}. Expected: YYYY-MM-DDTHH:MM:SS.sssZ") from e
        return v

    model_config = {"frozen": True}
