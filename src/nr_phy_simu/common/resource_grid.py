from __future__ import annotations

from nr_phy_simu.rx.frequency_extraction import FrequencyDomainExtractor
from nr_phy_simu.tx.resource_mapping import FrequencyDomainResourceMapper


# Backward-compatible aliases. New code should import from the dedicated
# tx/rx modules to keep stage ownership explicit.
NrResourceMapper = FrequencyDomainResourceMapper
DataExtractor = FrequencyDomainExtractor
