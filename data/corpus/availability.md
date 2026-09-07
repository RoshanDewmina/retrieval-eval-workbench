# Availability targets

The case search page has a monthly availability objective of 99.5 percent. Scheduled maintenance is announced at least forty-eight hours in advance and is excluded from that objective.

The ingestion worker retries a transient downstream failure up to three times using exponential backoff. A failed final attempt creates a case for General Review.

The service status page reports planned maintenance, active incidents, and the time of the last completed update.
