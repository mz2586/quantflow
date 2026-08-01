"""Market regime detection.

Classifies recent price action into a small set of regimes so a strategy can be told *when*
it is operating in conditions it was designed for. A trend-following strategy in a range
does not fail loudly — it bleeds slowly through repeated whipsaws, which is much harder to
notice than a crash.

Two detectors are provided:

- :class:`RuleBasedRegimeDetector` — transparent, deterministic, no training. The default,
  because an operator can read the thresholds and predict its output.
- :class:`GaussianMixtureRegimeDetector` — unsupervised clustering over the same features.
  More adaptive, but its cluster-to-regime mapping is inferred rather than declared, so it
  is opt-in.

Neither is allowed to place an order. A regime is *context* passed to the strategy, which
still produces the signal, which still passes through the risk engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from quantflow.core.errors import InsufficientDataError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO, safe_divide
from quantflow.domain.enums import MarketRegime
from quantflow.domain.market import Candle
from quantflow.strategy.indicators import ema, normalized_atr, stdev

logger = get_logger(__name__)

#: Bars required before a classification is attempted. Below this the features are too
#: noisy to mean anything, and the detector says UNKNOWN rather than guessing.
MIN_BARS = 60

#: Fractional distance between fast and slow trend lines above which price is "trending".
TREND_THRESHOLD = Decimal("0.02")

#: Normalised ATR above which volatility dominates whatever the trend is doing.
HIGH_VOLATILITY_THRESHOLD = Decimal("0.03")

#: Trailing bars used for the dispersion and directional-share features.
SHORT_WINDOW = 20


@dataclass(frozen=True, slots=True)
class RegimeFeatures:
    """The measurements a classification is made from.

    Returned alongside every classification so a surprising label can be explained rather
    than merely observed.
    """

    trend_strength: Decimal
    """Signed fractional gap between the fast and slow trend lines."""
    normalized_volatility: Decimal
    """ATR as a fraction of price."""
    return_dispersion: Decimal
    """Standard deviation of recent returns."""
    directional_share: Decimal
    """Fraction of recent bars that closed in the dominant direction."""
    price_position: Decimal
    """Where price sits within its recent range, 0 (low) to 1 (high)."""

    def to_dict(self) -> dict[str, float]:
        """Serialise for persistence and the API."""
        return {
            "trend_strength": float(self.trend_strength),
            "normalized_volatility": float(self.normalized_volatility),
            "return_dispersion": float(self.return_dispersion),
            "directional_share": float(self.directional_share),
            "price_position": float(self.price_position),
        }

    def as_vector(self) -> list[float]:
        """Feature vector for the clustering detector."""
        return [
            float(self.trend_strength),
            float(self.normalized_volatility),
            float(self.return_dispersion),
            float(self.directional_share),
            float(self.price_position),
        ]


@dataclass(frozen=True, slots=True)
class RegimeObservation:
    """One classification."""

    regime: MarketRegime
    confidence: Decimal
    timestamp: datetime
    features: RegimeFeatures
    detector: str
    reason: str = ""

    @property
    def is_confident(self) -> bool:
        """Whether the classification is strong enough to act on."""
        return self.confidence >= Decimal("0.6")

    def to_dict(self) -> dict[str, Any]:
        """Serialise for persistence and the API."""
        return {
            "regime": self.regime.value,
            "confidence": str(self.confidence),
            "timestamp": self.timestamp.isoformat(),
            "features": self.features.to_dict(),
            "detector": self.detector,
            "reason": self.reason,
        }


@runtime_checkable
class RegimeDetector(Protocol):
    """Classifies recent price action."""

    @property
    def name(self) -> str:
        """Detector identifier."""
        ...

    def detect(self, candles: Sequence[Candle]) -> RegimeObservation:
        """Classify the most recent bar's regime."""
        ...


def extract_features(candles: Sequence[Candle], *, lookback: int = 50) -> RegimeFeatures:
    """Compute regime features from recent bars.

    Raises:
        InsufficientDataError: if there are too few bars for the features to be stable.

    """
    if len(candles) < MIN_BARS:
        raise InsufficientDataError(
            f"regime detection needs {MIN_BARS} bars, got {len(candles)}",
            available=len(candles),
            required=MIN_BARS,
        )

    window = list(candles[-max(lookback, MIN_BARS) :])
    closes = [candle.close for candle in window]
    price = closes[-1]

    fast = ema(closes, 20)[-1]
    slow = ema(closes, 50)[-1]
    trend = safe_divide(fast - slow, price) if fast is not None and slow is not None else ZERO

    volatility = normalized_atr(window, 14)[-1] or ZERO

    returns = [safe_divide(current - previous, previous) for previous, current in pairwise(closes)]
    dispersion = (
        stdev(returns, min(SHORT_WINDOW, len(returns)))[-1]
        if len(returns) >= SHORT_WINDOW
        else ZERO
    )

    ups = sum(1 for value in returns[-SHORT_WINDOW:] if value > ZERO)
    downs = sum(1 for value in returns[-SHORT_WINDOW:] if value < ZERO)
    total = max(1, ups + downs)
    directional = Decimal(max(ups, downs)) / Decimal(total)

    high = max(candle.high for candle in window)
    low = min(candle.low for candle in window)
    position = safe_divide(price - low, high - low, default=Decimal("0.5"))

    return RegimeFeatures(
        trend_strength=trend,
        normalized_volatility=volatility,
        return_dispersion=dispersion or ZERO,
        directional_share=directional,
        price_position=position,
    )


class RuleBasedRegimeDetector:
    """Threshold classifier over the regime features.

    Deterministic and readable. An operator can look at the thresholds and predict the
    output, which matters more here than adaptiveness: a regime label that silently changes
    behaviour needs to be explicable after the fact.
    """

    __slots__ = ("_high_volatility", "_trend_threshold")

    def __init__(
        self,
        *,
        trend_threshold: Decimal = TREND_THRESHOLD,
        high_volatility: Decimal = HIGH_VOLATILITY_THRESHOLD,
    ) -> None:
        self._trend_threshold = trend_threshold
        self._high_volatility = high_volatility

    @property
    def name(self) -> str:
        """Detector identifier."""
        return "rule_based"

    def detect(self, candles: Sequence[Candle]) -> RegimeObservation:
        """Classify the most recent bar."""
        features = extract_features(candles)
        timestamp = candles[-1].close_time

        # Volatility is checked first: in a violent market the trend label is unreliable,
        # and sizing should shrink regardless of direction.
        if features.normalized_volatility >= self._high_volatility:
            confidence = min(
                ONE,
                safe_divide(features.normalized_volatility, self._high_volatility) / Decimal("2")
                + Decimal("0.5"),
            )
            return RegimeObservation(
                regime=MarketRegime.HIGH_VOLATILITY,
                confidence=confidence,
                timestamp=timestamp,
                features=features,
                detector=self.name,
                reason=(
                    f"ATR is {features.normalized_volatility:.2%} of price, at or above "
                    f"the {self._high_volatility:.2%} threshold"
                ),
            )

        if features.trend_strength >= self._trend_threshold:
            return RegimeObservation(
                regime=MarketRegime.BULL_TREND,
                confidence=self._trend_confidence(features),
                timestamp=timestamp,
                features=features,
                detector=self.name,
                reason=f"fast EMA is {features.trend_strength:.2%} above slow",
            )

        if features.trend_strength <= -self._trend_threshold:
            return RegimeObservation(
                regime=MarketRegime.BEAR_TREND,
                confidence=self._trend_confidence(features),
                timestamp=timestamp,
                features=features,
                detector=self.name,
                reason=f"fast EMA is {abs(features.trend_strength):.2%} below slow",
            )

        return RegimeObservation(
            regime=MarketRegime.RANGE,
            confidence=self._range_confidence(features),
            timestamp=timestamp,
            features=features,
            detector=self.name,
            reason=(
                f"trend strength {features.trend_strength:.2%} is inside "
                f"±{self._trend_threshold:.2%}"
            ),
        )

    def _trend_confidence(self, features: RegimeFeatures) -> Decimal:
        """Stronger separation and more one-sided bars mean a more confident trend."""
        magnitude = min(ONE, safe_divide(abs(features.trend_strength), self._trend_threshold * 3))
        return min(ONE, (magnitude + features.directional_share) / Decimal("2"))

    def _range_confidence(self, features: RegimeFeatures) -> Decimal:
        """A flatter trend and a more balanced up/down split mean a clearer range."""
        flatness = ONE - min(ONE, safe_divide(abs(features.trend_strength), self._trend_threshold))
        balance = ONE - abs(features.directional_share - Decimal("0.5")) * 2
        return min(ONE, max(ZERO, (flatness + balance) / Decimal("2")))


class GaussianMixtureRegimeDetector:
    """Unsupervised clustering over the regime features.

    Fitted on historical features, then each cluster is mapped to a regime by inspecting
    its centroid. More adaptive than fixed thresholds, but the mapping is *inferred*, so
    this is opt-in rather than the default.

    Falls back to the rule-based detector when scikit-learn is unavailable or the model has
    not been fitted — a missing optional dependency must degrade, not crash.
    """

    __slots__ = ("_fallback", "_labels", "_model", "_n_components")

    def __init__(self, *, n_components: int = 4) -> None:
        self._n_components = n_components
        self._model: Any = None
        self._labels: dict[int, MarketRegime] = {}
        self._fallback = RuleBasedRegimeDetector()

    @property
    def name(self) -> str:
        """Detector identifier."""
        return "gaussian_mixture"

    @property
    def is_fitted(self) -> bool:
        """Whether a model has been trained."""
        return self._model is not None

    def fit(self, candles: Sequence[Candle], *, step: int = 5) -> bool:
        """Fit the mixture model on a history of feature vectors.

        Returns:
            ``True`` if a model was fitted, ``False`` if the dependency or the data was
            insufficient — in which case detection silently uses the rule-based fallback.

        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            logger.warning("regime.sklearn_unavailable", detector=self.name)
            return False

        vectors: list[list[float]] = []
        for end in range(MIN_BARS, len(candles), step):
            try:
                vectors.append(extract_features(candles[:end]).as_vector())
            except InsufficientDataError:
                continue

        if len(vectors) < self._n_components * 5:
            logger.warning(
                "regime.insufficient_training_data",
                samples=len(vectors),
                required=self._n_components * 5,
            )
            return False

        model = GaussianMixture(
            n_components=self._n_components,
            covariance_type="full",
            # A fixed seed: an unsupervised model that relabels its clusters between runs
            # would silently change strategy behaviour on a restart.
            random_state=42,
            n_init=3,
        )
        model.fit(vectors)
        self._model = model
        self._labels = self._map_clusters(model)
        logger.info(
            "regime.model_fitted",
            samples=len(vectors),
            components=self._n_components,
            mapping={key: value.value for key, value in self._labels.items()},
        )
        return True

    def _map_clusters(self, model: Any) -> dict[int, MarketRegime]:
        """Assign a regime to each cluster by inspecting its centroid.

        Feature order is (trend, volatility, dispersion, directional share, position).
        """
        mapping: dict[int, MarketRegime] = {}
        for index, centre in enumerate(model.means_):
            trend, volatility = float(centre[0]), float(centre[1])
            if volatility >= float(HIGH_VOLATILITY_THRESHOLD):
                mapping[index] = MarketRegime.HIGH_VOLATILITY
            elif trend >= float(TREND_THRESHOLD):
                mapping[index] = MarketRegime.BULL_TREND
            elif trend <= -float(TREND_THRESHOLD):
                mapping[index] = MarketRegime.BEAR_TREND
            else:
                mapping[index] = MarketRegime.RANGE
        return mapping

    def detect(self, candles: Sequence[Candle]) -> RegimeObservation:
        """Classify the most recent bar, falling back when unfitted."""
        if not self.is_fitted:
            return self._fallback.detect(candles)

        features = extract_features(candles)
        vector = [features.as_vector()]
        cluster = int(self._model.predict(vector)[0])
        probabilities = self._model.predict_proba(vector)[0]
        confidence = Decimal(str(round(float(probabilities[cluster]), 4)))

        return RegimeObservation(
            regime=self._labels.get(cluster, MarketRegime.UNKNOWN),
            confidence=confidence,
            timestamp=candles[-1].close_time,
            features=features,
            detector=self.name,
            reason=f"cluster {cluster} with probability {confidence:.2f}",
        )


def build_detector(kind: str = "rule_based", **kwargs: Any) -> RegimeDetector:
    """Construct a detector by name.

    Raises:
        ValidationError: for an unknown detector.

    """
    from quantflow.core.errors import ValidationError

    if kind == "rule_based":
        return RuleBasedRegimeDetector(**kwargs)
    if kind == "gaussian_mixture":
        return GaussianMixtureRegimeDetector(**kwargs)
    raise ValidationError(
        f"unknown regime detector {kind!r}; available: rule_based, gaussian_mixture"
    )


def regime_history(
    candles: Sequence[Candle], detector: RegimeDetector, *, step: int = 1
) -> list[RegimeObservation]:
    """Classify a whole series, for charting and for backtest attribution."""
    observations: list[RegimeObservation] = []
    for end in range(MIN_BARS, len(candles) + 1, step):
        try:
            observations.append(detector.detect(candles[:end]))
        except InsufficientDataError:
            continue
    return observations


def summarise(observations: Sequence[RegimeObservation]) -> dict[str, Any]:
    """Distribution of regimes across a series."""
    if not observations:
        return {"total": 0, "distribution": {}}
    counts: dict[str, int] = {}
    for observation in observations:
        counts[observation.regime.value] = counts.get(observation.regime.value, 0) + 1
    total = len(observations)
    return {
        "total": total,
        "distribution": {key: round(value / total, 4) for key, value in counts.items()},
        "current": observations[-1].regime.value,
        "current_confidence": str(observations[-1].confidence),
    }
