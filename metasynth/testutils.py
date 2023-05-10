"""Testing utilities for plugins."""


from __future__ import annotations

import jsonschema
import polars as pl
from jsonschema.exceptions import SchemaError

from metasynth.dataset import _jsonify
from metasynth.distribution.base import BaseDistribution
from metasynth.privacy import BasePrivacy
from metasynth.provider import (get_distribution_provider,
                                BaseDistributionProvider,
                                DistributionProviderList)


def check_distribution_provider(provider_name: str):
    """Check internal consistency of a distribution provider.

    Arguments
    ---------
    provider_name:
        Name of the provider to be tested.
    """
    provider = get_distribution_provider(provider_name)
    print(type(provider))
    assert isinstance(provider, BaseDistributionProvider)
    assert len(provider.distributions) > 0
    assert all(issubclass(dist, BaseDistribution) for dist in provider.distributions)
    assert isinstance(provider.name, str)
    assert len(provider.name) > 0
    assert provider.name == provider_name
    assert isinstance(provider.version, str)
    assert len(provider.version) > 0


def check_distribution(distribution: type[BaseDistribution], privacy: BasePrivacy,
                       provenance: str):
    """Check whether the distributions in the package can be validated positively.

    Arguments
    ---------
    distribution:
        Distribution to validate to check whether it behaves as expected.
    privacy:
        Level/type of privacy the distribution adheres to.
    provenance:
        Which provider/plugin/package provides the distribution.
    """
    # Check the schema of the distribution.
    schema = distribution.schema()
    dist_dict = distribution.default_distribution().to_dict()
    try:
        jsonschema.validate(_jsonify(dist_dict), schema)
    except SchemaError as err:
        raise ValueError(f"Failed distribution validation for {distribution.__name__}") from err

    # Check the privacy
    assert privacy.is_compatible(distribution)
    DistributionProviderList(provenance).find_distribution(distribution.implements, privacy)

    assert len(distribution.implements.split(".")) == 2
    assert distribution.provenance == provenance
    assert distribution.var_type != "unknown"
    dist = distribution.default_distribution()
    series = pl.Series([dist.draw() for _ in range(100)])
    new_dist = distribution.fit(series, **privacy.fit_kwargs)
    assert isinstance(new_dist, distribution)
    assert set(list(new_dist.to_dict())) >= set(
        ("implements", "provenance", "class_name", "parameters"))
